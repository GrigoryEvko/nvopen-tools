// Function: sub_1B932A0
// Address: 0x1b932a0
//
__int64 __fastcall sub_1B932A0(__int64 a1, int *a2, __int64 a3)
{
  bool v3; // zf
  int *v4; // r13
  __int64 v5; // r12
  unsigned int v6; // r14d
  unsigned int v7; // ebx
  int v9; // [rsp+8h] [rbp-38h] BYREF
  _DWORD v10[13]; // [rsp+Ch] [rbp-34h] BYREF

  v3 = *(_QWORD *)(a1 + 16) == 0;
  v9 = *a2;
  if ( v3 )
LABEL_9:
    sub_4263D6(a1, a2, a3);
  v4 = a2;
  v5 = a1;
  a2 = &v9;
  v6 = (*(__int64 (__fastcall **)(__int64, int *))(a1 + 24))(a1, &v9);
  v7 = 2 * *v4;
  if ( v7 < v4[1] )
  {
    while ( 1 )
    {
      v3 = *(_QWORD *)(v5 + 16) == 0;
      v10[0] = v7;
      if ( v3 )
        goto LABEL_9;
      a2 = v10;
      a1 = v5;
      if ( (_BYTE)v6 != (*(unsigned __int8 (__fastcall **)(__int64, _DWORD *))(v5 + 24))(v5, v10) )
        break;
      v7 *= 2;
      if ( v4[1] <= v7 )
        return v6;
    }
    v4[1] = v7;
  }
  return v6;
}
