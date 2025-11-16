// Function: sub_2BE0770
// Address: 0x2be0770
//
__int64 __fastcall sub_2BE0770(__int64 a1)
{
  int v2; // eax
  unsigned int v3; // r13d
  unsigned __int64 v5; // rdx
  unsigned __int64 *v6; // r14
  __int64 v7; // rax
  unsigned __int64 v8; // r15
  __int64 v9; // rbx
  char v10; // di
  int v11; // eax
  char v12; // r8
  unsigned __int64 v13; // r15
  __int64 v14; // rbx
  char v15; // di
  int v16; // eax

  v2 = *(_DWORD *)(a1 + 152);
  if ( v2 == 2 )
  {
    v3 = sub_2BE0030(a1);
    if ( (_BYTE)v3 )
    {
      v5 = *(_QWORD *)(a1 + 280);
      v6 = (unsigned __int64 *)(a1 + 272);
      if ( v5 )
      {
        v7 = 0;
        v8 = 0;
        do
        {
          v9 = 8 * v7;
          v10 = *(_BYTE *)(*(_QWORD *)(a1 + 272) + v8++);
          v11 = sub_2BDCD80(v10, 8);
          v5 = *(_QWORD *)(a1 + 280);
          v7 = v9 + v11;
        }
        while ( v5 > v8 );
        goto LABEL_13;
      }
LABEL_19:
      v12 = 0;
      goto LABEL_14;
    }
    v2 = *(_DWORD *)(a1 + 152);
  }
  if ( v2 != 3 )
    goto LABEL_3;
  v3 = sub_2BE0030(a1);
  if ( (_BYTE)v3 )
  {
    v5 = *(_QWORD *)(a1 + 280);
    v6 = (unsigned __int64 *)(a1 + 272);
    if ( v5 )
    {
      v7 = 0;
      v13 = 0;
      do
      {
        v14 = 16 * v7;
        v15 = *(_BYTE *)(*(_QWORD *)(a1 + 272) + v13++);
        v16 = sub_2BDCD80(v15, 16);
        v5 = *(_QWORD *)(a1 + 280);
        v7 = v14 + v16;
      }
      while ( v5 > v13 );
LABEL_13:
      v12 = v7;
LABEL_14:
      sub_2240FD0(v6, 0, v5, 1u, v12);
      return v3;
    }
    goto LABEL_19;
  }
  v2 = *(_DWORD *)(a1 + 152);
LABEL_3:
  v3 = 0;
  if ( v2 != 1 )
    return v3;
  return sub_2BE0030(a1);
}
