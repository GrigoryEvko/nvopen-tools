// Function: sub_2F418E0
// Address: 0x2f418e0
//
__int64 __fastcall sub_2F418E0(__int64 a1, __int64 a2, __int64 a3, unsigned __int16 ***a4)
{
  __int64 *v6; // r14
  __int64 v7; // rax
  __int64 v8; // rcx
  unsigned int *v9; // rax
  __int64 v10; // rdx
  unsigned __int16 *v12; // r12
  __int64 v13; // r15
  __int64 v14; // r14
  __int64 v15; // r15
  __int64 v16; // r14
  __int64 v17; // [rsp+0h] [rbp-B0h]
  __int64 v18; // [rsp+8h] [rbp-A8h]
  __m128i v19; // [rsp+10h] [rbp-A0h] BYREF
  _QWORD v20[4]; // [rsp+20h] [rbp-90h] BYREF
  char v21; // [rsp+40h] [rbp-70h]
  char v22; // [rsp+41h] [rbp-6Fh]
  _QWORD v23[4]; // [rsp+50h] [rbp-60h] BYREF
  char v24; // [rsp+70h] [rbp-40h]
  char v25; // [rsp+71h] [rbp-3Fh]

  v6 = (__int64 *)sub_2E88D60(a3);
  v7 = v6[43];
  v8 = v7 & 0x400;
  if ( (v7 & 0x400) == 0 )
  {
    BYTE1(v7) |= 4u;
    v6[43] = v7;
  }
  v9 = (unsigned int *)(*(_QWORD *)(a1 + 32) + 24LL * *((unsigned __int16 *)*a4 + 12));
  v10 = *v9;
  if ( *(_DWORD *)(a1 + 40) != (_DWORD)v10 )
  {
    v17 = *(_QWORD *)(a1 + 32) + 24LL * *((unsigned __int16 *)*a4 + 12);
    v18 = v8;
    sub_2F60630(a1 + 32, a4, v10, v8);
    v9 = (unsigned int *)v17;
    v8 = v18;
  }
  if ( v9[1] )
  {
    v12 = (unsigned __int16 *)*((_QWORD *)v9 + 2);
    if ( *(_BYTE *)(a2 + 16) != 1 && !v8 )
    {
      if ( (unsigned int)*(unsigned __int16 *)(a3 + 68) - 1 > 1 )
      {
        v15 = **(_QWORD **)(*(_QWORD *)(a1 + 384) + 32LL);
        v16 = sub_B2BE50(v15);
        sub_B157E0((__int64)&v19, (_QWORD *)(a3 + 56));
        v22 = 1;
        v20[0] = "ran out of registers during register allocation";
        v21 = 3;
        sub_B158E0((__int64)v23, (__int64)v20, v15, &v19, 0);
        sub_B6EB20(v16, (__int64)v23);
      }
      else
      {
        v25 = 1;
        v23[0] = "inline assembly requires more registers than available";
        v24 = 3;
        sub_2E8D9C0(a3, (__int64)v23);
      }
    }
    return *v12;
  }
  else
  {
    if ( !v8 )
    {
      v13 = *v6;
      v14 = sub_B2BE50(*v6);
      sub_B157E0((__int64)&v19, (_QWORD *)(a3 + 56));
      v22 = 1;
      v20[0] = "no registers from class available to allocate";
      v21 = 3;
      sub_B158E0((__int64)v23, (__int64)v20, v13, &v19, 0);
      sub_B6EB20(v14, (__int64)v23);
    }
    return ***a4;
  }
}
