// Function: sub_31AFDC0
// Address: 0x31afdc0
//
__int64 __fastcall sub_31AFDC0(__int64 a1, _QWORD *a2, __int64 a3, unsigned int a4)
{
  _QWORD *v4; // r14
  _QWORD *v6; // r15
  __int64 v7; // rax
  __int64 v8; // r8
  __int64 v9; // rdx
  __int64 v11; // [rsp+0h] [rbp-60h]
  _BYTE v12[80]; // [rsp+10h] [rbp-50h] BYREF

  v4 = &a2[a3];
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x400000000LL;
  if ( a2 != v4 )
  {
    v6 = a2;
    do
    {
      (*(void (__fastcall **)(_BYTE *, _QWORD, _QWORD, __int64))(*(_QWORD *)*v6 + 16LL))(v12, *v6, a4, 1);
      v7 = sub_318E5D0((__int64)v12);
      v9 = *(unsigned int *)(a1 + 8);
      if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
      {
        v11 = v7;
        sub_C8D5F0(a1, (const void *)(a1 + 16), v9 + 1, 8u, v8, v9 + 1);
        v9 = *(unsigned int *)(a1 + 8);
        v7 = v11;
      }
      ++v6;
      *(_QWORD *)(*(_QWORD *)a1 + 8 * v9) = v7;
      ++*(_DWORD *)(a1 + 8);
    }
    while ( v4 != v6 );
  }
  return a1;
}
