// Function: sub_3990E70
// Address: 0x3990e70
//
__int64 __fastcall sub_3990E70(__int64 a1, __int64 a2)
{
  unsigned int v2; // r15d
  __int64 v4; // rbx
  __int64 v5; // rcx
  unsigned int v6; // r8d
  int v7; // r9d
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rbx
  unsigned int v11; // [rsp+18h] [rbp-68h]
  unsigned int v12; // [rsp+1Ch] [rbp-64h]
  int v13; // [rsp+20h] [rbp-60h]
  int v14; // [rsp+28h] [rbp-58h]
  __int64 v15; // [rsp+30h] [rbp-50h] BYREF
  __int64 v16; // [rsp+38h] [rbp-48h]
  unsigned __int8 v17; // [rsp+40h] [rbp-40h]

  if ( *(_QWORD *)a1 == *(_QWORD *)a2
    && (v4 = **(_QWORD **)(a2 + 16),
        sub_15B1350(
          (__int64)&v15,
          *(unsigned __int64 **)(**(_QWORD **)(a1 + 16) + 24LL),
          *(unsigned __int64 **)(**(_QWORD **)(a1 + 16) + 32LL)),
        v17)
    && (sub_15B1350((__int64)&v15, *(unsigned __int64 **)(v4 + 24), *(unsigned __int64 **)(v4 + 32)), (v2 = v17) != 0) )
  {
    v8 = *(unsigned int *)(a2 + 24);
    if ( *(_DWORD *)(a1 + 24) )
    {
      v11 = 0;
      v6 = 0;
      v9 = 0;
      do
      {
        if ( v6 < (unsigned int)v8 )
        {
          v12 = v6;
          v10 = **(_QWORD **)(a2 + 16);
          sub_15B1350(
            (__int64)&v15,
            *(unsigned __int64 **)(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 32 * v9) + 24LL),
            *(unsigned __int64 **)(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 32 * v9) + 32LL));
          v14 = v15;
          v13 = v16;
          sub_15B1350((__int64)&v15, *(unsigned __int64 **)(v10 + 24), *(unsigned __int64 **)(v10 + 32));
          v6 = v12;
          if ( (unsigned int)v16 < v13 + v14 )
            return 0;
          v8 = *(unsigned int *)(a2 + 24);
        }
        v9 = ++v11;
      }
      while ( *(_DWORD *)(a1 + 24) > v11 );
    }
    sub_3990CF0(a1, *(const void **)(a2 + 16), v8, v5, v6, v7);
    *(_QWORD *)(a1 + 8) = *(_QWORD *)(a2 + 8);
  }
  else
  {
    return 0;
  }
  return v2;
}
