// Function: sub_3305470
// Address: 0x3305470
//
__int64 __fastcall sub_3305470(__int64 **a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 *v9; // rdi
  __int64 v10; // rdx
  __int64 *v12; // rdi
  __int64 v16; // [rsp+20h] [rbp-40h] BYREF
  __int64 v17; // [rsp+28h] [rbp-38h]
  __int64 v18; // [rsp+30h] [rbp-30h]
  __int64 v19; // [rsp+38h] [rbp-28h]
  __int64 v20; // [rsp+40h] [rbp-20h]
  __int64 v21; // [rsp+48h] [rbp-18h]

  if ( (*(_WORD *)(a2 + 32) & 0x380) != 0 )
  {
    if ( !(_BYTE)qword_5038248 )
      return 0;
    v7 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 80LL);
    if ( *(_DWORD *)(v7 + 24) == 35 && (*(_BYTE *)(v7 + 32) & 8) != 0 )
    {
      return 0;
    }
    else
    {
      v8 = sub_3265190(*a1, a2, a3, a4, a5, a6);
      v9 = *a1;
      v16 = a3;
      v19 = v10;
      v17 = a4;
      v20 = a5;
      v18 = v8;
      v21 = a6;
      return sub_32EB790((__int64)v9, a2, &v16, 3, 1);
    }
  }
  else
  {
    v12 = *a1;
    v16 = a3;
    LODWORD(v17) = a4;
    v18 = a5;
    LODWORD(v19) = a6;
    return sub_32EB790((__int64)v12, a2, &v16, 2, 1);
  }
}
