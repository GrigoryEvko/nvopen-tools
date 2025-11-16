// Function: sub_35B51A0
// Address: 0x35b51a0
//
__int64 __fastcall sub_35B51A0(__int64 a1, unsigned __int16 ***a2, __int64 a3)
{
  __int64 *v4; // rdx
  __int64 v5; // rax
  __int64 v6; // r14
  __int64 v7; // r15
  __int64 v8; // rax
  unsigned __int16 *v9; // rbx
  const char *v10; // rax
  __int64 v12; // [rsp+0h] [rbp-B0h]
  __int64 v13; // [rsp+8h] [rbp-A8h]
  __m128i v14; // [rsp+10h] [rbp-A0h] BYREF
  const char *v15; // [rsp+20h] [rbp-90h] BYREF
  char v16; // [rsp+40h] [rbp-70h]
  char v17; // [rsp+41h] [rbp-6Fh]
  _QWORD v18[4]; // [rsp+50h] [rbp-60h] BYREF
  char v19; // [rsp+70h] [rbp-40h]
  char v20; // [rsp+71h] [rbp-3Fh]

  v4 = *(__int64 **)(*(_QWORD *)(a1 + 24) + 24LL);
  v5 = v4[43];
  v6 = v5 & 0x400;
  if ( (v5 & 0x400) == 0 )
  {
    BYTE1(v5) |= 4u;
    v4[43] = v5;
  }
  v7 = *v4;
  v13 = sub_B2BE50(*v4);
  v8 = *(_QWORD *)(a1 + 48) + 24LL * *((unsigned __int16 *)*a2 + 12);
  if ( *(_DWORD *)(a1 + 56) != *(_DWORD *)v8 )
  {
    v12 = *(_QWORD *)(a1 + 48) + 24LL * *((unsigned __int16 *)*a2 + 12);
    sub_2F60630(a1 + 48, a2);
    v8 = v12;
  }
  if ( *(_DWORD *)(v8 + 4) )
  {
    v9 = *(unsigned __int16 **)(v8 + 16);
    if ( v6 )
      return *v9;
    if ( a3 )
    {
      if ( (unsigned int)*(unsigned __int16 *)(a3 + 68) - 1 <= 1 )
      {
        v20 = 1;
        v18[0] = "inline assembly requires more registers than available";
        v19 = 3;
        sub_2E8D9C0(a3, (__int64)v18);
        return *v9;
      }
      sub_B157E0((__int64)&v14, (_QWORD *)(a3 + 56));
    }
    else
    {
      v14 = 0u;
    }
    v17 = 1;
    v10 = "ran out of registers during register allocation";
    goto LABEL_10;
  }
  v9 = **a2;
  if ( !v6 )
  {
    if ( a3 )
      sub_B157E0((__int64)&v14, (_QWORD *)(a3 + 56));
    else
      v14 = 0u;
    v17 = 1;
    v10 = "no registers from class available to allocate";
LABEL_10:
    v15 = v10;
    v16 = 3;
    sub_B158E0((__int64)v18, (__int64)&v15, v7, &v14, 0);
    sub_B6EB20(v13, (__int64)v18);
  }
  return *v9;
}
