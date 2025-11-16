// Function: sub_AC64E0
// Address: 0xac64e0
//
__int64 __fastcall sub_AC64E0(__int64 a1, __int64 a2, _QWORD *a3)
{
  int v4; // r12d
  __int64 v6; // r14
  int v7; // r12d
  int v8; // eax
  unsigned int v9; // ecx
  int v10; // r10d
  __int64 v11; // r9
  unsigned int i; // r15d
  __int64 v13; // rdx
  int v14; // r8d
  unsigned int v15; // r15d
  char v16; // al
  unsigned int v17; // [rsp+4h] [rbp-4Ch]
  int v18; // [rsp+8h] [rbp-48h]
  int v19; // [rsp+Ch] [rbp-44h]
  __int64 v20; // [rsp+10h] [rbp-40h]

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    *a3 = 0;
    return 0;
  }
  v6 = *(_QWORD *)(a1 + 8);
  v7 = v4 - 1;
  v8 = sub_C4F140(a2);
  v9 = *(_DWORD *)(a2 + 8);
  v10 = 1;
  v11 = 0;
  for ( i = v7 & v8; ; i = v7 & v15 )
  {
    v13 = v6 + 24LL * i;
    v14 = *(_DWORD *)(v13 + 8);
    if ( v14 == v9 )
    {
      if ( v9 <= 0x40 )
      {
        if ( *(_QWORD *)a2 == *(_QWORD *)v13 )
        {
LABEL_11:
          *a3 = v13;
          return 1;
        }
      }
      else
      {
        v17 = v9;
        v18 = *(_DWORD *)(v13 + 8);
        v19 = v10;
        v20 = v11;
        v16 = sub_C43C50(a2, v6 + 24LL * i);
        v13 = v6 + 24LL * i;
        if ( v16 )
          goto LABEL_11;
        v11 = v20;
        v10 = v19;
        v14 = v18;
        v9 = v17;
      }
    }
    if ( !v14 )
      break;
LABEL_6:
    v15 = v10 + i;
    ++v10;
  }
  if ( *(_QWORD *)v13 != -1 )
  {
    if ( !v11 && *(_QWORD *)v13 == -2 )
      v11 = v13;
    goto LABEL_6;
  }
  if ( !v11 )
    v11 = v13;
  *a3 = v11;
  return 0;
}
