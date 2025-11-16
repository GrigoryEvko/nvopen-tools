// Function: sub_256F570
// Address: 0x256f570
//
__int64 __fastcall sub_256F570(__int64 a1, __int64 a2, __int64 a3, unsigned __int8 *a4, unsigned __int8 a5)
{
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  unsigned __int64 *v11; // rax
  unsigned __int64 *v12; // r13
  unsigned __int8 *v13; // r14
  unsigned __int8 *v15; // r15
  unsigned __int64 v16; // rbx
  __int64 v17; // rcx
  __int64 v18; // rsi
  __int64 v19; // r8
  __int64 v20; // r9
  unsigned __int64 v21; // rbx
  __int64 v22[3]; // [rsp+0h] [rbp-50h] BYREF
  __int64 v23[7]; // [rsp+18h] [rbp-38h] BYREF

  v22[0] = a2;
  v22[1] = a3;
  if ( (unsigned __int8)sub_2509800(v22) == 7 )
  {
    v16 = sub_2509740(v22);
    v18 = (unsigned int)sub_250CB50(v22, 0);
    if ( (*(_BYTE *)(v16 + 7) & 0x40) != 0 )
      v21 = *(_QWORD *)(v16 - 8);
    else
      v21 = v16 - 32LL * (*(_DWORD *)(v16 + 4) & 0x7FFFFFF);
    return sub_256E5A0(a1, v21 + 32 * v18, a4, v17, v19, v20);
  }
  else
  {
    v23[0] = sub_250D070(v22);
    v11 = (unsigned __int64 *)sub_256F330(a1 + 1632, v23, v7, v8, v9, v10);
    v12 = v11;
    v13 = (unsigned __int8 *)(*v11 & 0xFFFFFFFFFFFFFFF8LL);
    if ( v13
      && ((v15 = sub_BD3990((unsigned __int8 *)(*v11 & 0xFFFFFFFFFFFFFFF8LL), (__int64)v23),
           v15 == sub_BD3990(a4, (__int64)v23))
       || (unsigned int)*v13 - 12 <= 1) )
    {
      return 0;
    }
    else
    {
      *v12 = (4LL * a5) | (unsigned __int64)a4 & 0xFFFFFFFFFFFFFFFBLL;
      return 1;
    }
  }
}
