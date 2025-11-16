// Function: sub_258F980
// Address: 0x258f980
//
__int64 __fastcall sub_258F980(__int64 *a1, __int64 a2)
{
  __int64 v2; // rbx
  _BYTE *v3; // r14
  __int64 v4; // r9
  __m128i v5; // rax
  unsigned __int64 v6; // rax
  __int64 v7; // rdx
  unsigned __int64 v9; // rax
  __int64 v10; // [rsp+0h] [rbp-90h]
  char v11; // [rsp+1Fh] [rbp-71h] BYREF
  unsigned __int64 v12; // [rsp+20h] [rbp-70h]
  __int64 v13; // [rsp+28h] [rbp-68h]
  __m128i v14[6]; // [rsp+30h] [rbp-60h] BYREF

  v2 = a1[1];
  if ( (*(_DWORD *)(a2 + 4) & 0x7FFFFFF) != 0 )
    v3 = *(_BYTE **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  else
    v3 = 0;
  v4 = *a1;
  v11 = 0;
  v10 = v4;
  v5.m128i_i64[0] = sub_250D2C0((unsigned __int64)v3, 0);
  v14[0] = v5;
  v6 = sub_2527850(v10, v14, v2, &v11, 2u);
  v12 = v6;
  v13 = v7;
  if ( v11 )
    goto LABEL_7;
  if ( !(_BYTE)v7 )
  {
LABEL_8:
    sub_BED950((__int64)v14, v2 + 104, a2);
    return 1;
  }
  if ( v6 )
  {
    v3 = (_BYTE *)v6;
LABEL_7:
    if ( (unsigned int)(unsigned __int8)*v3 - 12 <= 1 )
      goto LABEL_8;
    if ( *v3 == 20 )
    {
      v9 = sub_25096F0((_QWORD *)(a1[1] + 72));
      sub_250D230((unsigned __int64 *)v14, v9, 2, 0);
      sub_258F340((_QWORD *)*a1, a1[1], v14, 2, &v11, 0, 0);
      if ( v11 )
        sub_BED950((__int64)v14, a1[1] + 104, a2);
    }
  }
  return 1;
}
