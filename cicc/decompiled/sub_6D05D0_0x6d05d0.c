// Function: sub_6D05D0
// Address: 0x6d05d0
//
_DWORD *__fastcall sub_6D05D0(__int64 a1, int a2, int a3)
{
  _QWORD *v5; // rax
  _QWORD *i; // rdx
  __int64 v7; // r13
  __int64 v8; // rdi
  _DWORD *result; // rax
  __int64 v10; // r12
  __int64 v11; // rdx
  __int64 v12; // r14
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // [rsp+8h] [rbp-238h] BYREF
  char v16[17]; // [rsp+10h] [rbp-230h] BYREF
  char v17; // [rsp+21h] [rbp-21Fh]
  char v18; // [rsp+24h] [rbp-21Ch]
  _QWORD *v19; // [rsp+88h] [rbp-1B8h]
  _BYTE v20[16]; // [rsp+B0h] [rbp-190h] BYREF
  char v21; // [rsp+C0h] [rbp-180h]
  __int64 v22; // [rsp+FCh] [rbp-144h]
  __int64 v23; // [rsp+140h] [rbp-100h]

  sub_6E6260(v20);
  if ( dword_4F077BC && !dword_4D047B8 )
    sub_6256B0(1);
  sub_6E1DD0(&v15);
  sub_6E1E00(4, v16, 1, 0);
  v17 |= 0xC0u;
  if ( a3 )
    v18 |= 2u;
  if ( a1 )
  {
    v5 = *(_QWORD **)(a1 + 56);
    for ( i = (_QWORD *)(a1 + 56); v5; v5 = (_QWORD *)*v5 )
      i = v5;
    v19 = i;
  }
  if ( dword_4D04428 && word_4F06418[0] == 73 )
  {
    v14 = sub_6BA760(0, 0);
    sub_6E9FE0(v14, v20);
  }
  else
  {
    sub_69ED20((__int64)v20, 0, 0, 1);
  }
  if ( a1 )
  {
    if ( v21 == 5 )
    {
      v12 = v23;
      sub_848800(v23, a1, 0, 310, v20);
      sub_6E1990(v12);
    }
    else if ( !dword_4F077BC
           || ((unsigned __int64)(qword_4F077A8 - 30400LL) > 0x257F || !a2)
           && dword_4F04C44 == -1
           && (v11 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v11 + 6) & 6) == 0)
           && *(_BYTE *)(v11 + 4) != 12
           || (unsigned __int8)(v21 - 1) > 1u )
    {
      sub_843D70(v20, a1, 0, 310);
    }
    v7 = sub_6F6F40(v20, 0);
    if ( !dword_4F07590 )
    {
      if ( dword_4F04C44 != -1
        || (v13 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v13 + 6) & 6) != 0)
        || *(_BYTE *)(v13 + 4) == 12 )
      {
        sub_6E2A90();
      }
    }
    v8 = v7;
    *(_QWORD *)(a1 + 40) = sub_6E2700(v7);
    *(_QWORD *)(qword_4D03C50 + 120LL) = 0;
    if ( !unk_4F04C50 )
    {
      v8 = a1;
      sub_85E780(a1);
    }
  }
  else
  {
    if ( v21 != 5 )
      sub_6F69D0(v20, 0);
    v10 = sub_6F6F40(v20, 0);
    sub_6E2A90();
    v8 = v10;
    sub_6E2700(v10);
  }
  sub_6E2B30(v8, 0);
  sub_6E1DF0(v15);
  result = &dword_4F061D8;
  *(_QWORD *)&dword_4F061D8 = v22;
  if ( dword_4F077BC )
  {
    result = &dword_4D047B8;
    if ( !dword_4D047B8 )
      return (_DWORD *)sub_6256B0(0);
  }
  return result;
}
