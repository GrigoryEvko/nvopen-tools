// Function: sub_73F570
// Address: 0x73f570
//
__int64 __fastcall sub_73F570(__int64 a1, __int64 a2, int a3, unsigned int a4, int a5)
{
  _QWORD *v5; // rbx
  __int64 v6; // r12
  __int64 v9; // rax
  int v10; // r8d
  __int64 v11; // rax
  __int64 v13; // [rsp+0h] [rbp-40h]
  int v15; // [rsp+Ch] [rbp-34h]

  if ( !a2 )
    return 0;
  v5 = (_QWORD *)a2;
  v6 = 0;
  if ( (*(_BYTE *)(a2 + 33) & 1) == 0 && (*(_WORD *)(a2 + 32) & 0x204) != 0x200 )
  {
    v9 = sub_73F4A0(a1, a2, a3, a4, a5);
    v10 = a5;
    v13 = v9;
    v6 = v9;
    while ( 1 )
    {
      v5 = (_QWORD *)*v5;
      if ( !v5 )
        break;
      v15 = v10;
      v11 = sub_73F4A0(a1, (__int64)v5, a3, a4, v10);
      v10 = v15;
      if ( v6 )
        *(_QWORD *)(v13 + 16) = v11;
      else
        v6 = v11;
      v13 = v11;
    }
  }
  return v6;
}
