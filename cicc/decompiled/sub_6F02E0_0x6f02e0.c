// Function: sub_6F02E0
// Address: 0x6f02e0
//
__int64 __fastcall sub_6F02E0(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, _BOOL4 *a5, __int64 a6)
{
  int v7; // r13d
  _QWORD *v9; // rax
  __int64 v10; // r9
  __int64 v11; // rcx
  _QWORD *v12; // r8
  __int64 v13; // rdx
  __int64 v14; // r14
  int v15; // eax
  _BOOL4 v16; // r15d
  const __m128i *v17; // rax
  char v19; // al
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // [rsp-8h] [rbp-138h]
  _QWORD *v23; // [rsp+8h] [rbp-128h]
  _QWORD *v24; // [rsp+8h] [rbp-128h]
  unsigned int v25; // [rsp+1Ch] [rbp-114h] BYREF
  const __m128i *v26; // [rsp+20h] [rbp-110h] BYREF
  const __m128i *v27; // [rsp+28h] [rbp-108h] BYREF
  _QWORD v28[32]; // [rsp+30h] [rbp-100h] BYREF

  v7 = a4;
  v25 = 0;
  v26 = (const __m128i *)sub_724DC0(a1, a2, a3, a4, a5, a6);
  v27 = 0;
  v9 = (_QWORD *)sub_6EFFF0(a1, a2, a3, 0, v26, (__int64 *)&v27, &v25);
  v11 = v25;
  v12 = v9;
  v13 = v22;
  if ( v25 )
  {
    v16 = 1;
    v14 = 0;
LABEL_4:
    if ( v12 )
    {
      v24 = v12;
      if ( (_QWORD *)a1 != v12 )
      {
        sub_76C7C0(v28, a2, v13, v11, v12, v10);
        v28[0] = sub_6DEBF0;
        v28[1] = sub_6E01F0;
        sub_76CDC0(v24);
      }
    }
    goto LABEL_10;
  }
  if ( v9 )
  {
    v23 = v9;
    v14 = *v9;
    v15 = sub_731B40(v9, a2, v22, v25);
    v12 = v23;
    v16 = v15 == 0;
    if ( v7 )
    {
      v19 = *((_BYTE *)v23 + 25);
      if ( (v19 & 1) != 0 )
      {
        v21 = sub_72D600(v14);
        v12 = v23;
        v14 = v21;
      }
      else if ( (v19 & 2) != 0 )
      {
        v20 = sub_72D6A0(v14);
        v12 = v23;
        v14 = v20;
      }
    }
    goto LABEL_4;
  }
  v17 = v27;
  if ( !v27 )
  {
    v17 = v26;
    v27 = v26;
  }
  v14 = v17[8].m128i_i64[0];
  v16 = 1;
LABEL_10:
  sub_724E30(&v26);
  *a5 = v16;
  return v14;
}
