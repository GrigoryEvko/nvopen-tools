// Function: sub_36E3A70
// Address: 0x36e3a70
//
__int64 __fastcall sub_36E3A70(__int64 *a1, __int64 a2)
{
  unsigned int v2; // r13d
  __int64 v3; // rax
  __int64 v5; // r13
  __int64 v8; // r12
  const char *v9; // r14
  size_t v10; // rax
  size_t v11; // r11
  _QWORD *v12; // rdx
  char *v13; // rax
  __int64 v14; // rax
  __int64 v15; // r9
  __int64 v16; // rsi
  _QWORD *v17; // r12
  unsigned __int64 v18; // rcx
  __int64 v19; // rax
  __int64 v20; // r8
  unsigned __int64 v21; // rdx
  __int64 v22; // r12
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  _QWORD *v26; // rax
  _QWORD *v27; // rdi
  size_t n; // [rsp+8h] [rbp-B8h]
  size_t na; // [rsp+8h] [rbp-B8h]
  __int64 v30; // [rsp+20h] [rbp-A0h]
  __int64 v31; // [rsp+30h] [rbp-90h] BYREF
  int v32; // [rsp+38h] [rbp-88h]
  _QWORD *v33; // [rsp+40h] [rbp-80h] BYREF
  size_t v34; // [rsp+48h] [rbp-78h]
  _QWORD v35[2]; // [rsp+50h] [rbp-70h] BYREF
  unsigned __int64 v36[2]; // [rsp+60h] [rbp-60h] BYREF
  __m128i v37; // [rsp+70h] [rbp-50h]
  __m128i v38; // [rsp+80h] [rbp-40h]

  v2 = 0;
  v3 = a1[121];
  if ( !v3 || *(_DWORD *)(v3 + 24) != 37 )
    return v2;
  v5 = *(_QWORD *)(v3 + 96);
  v8 = sub_3936750();
  v2 = sub_314C600(v5, v8);
  if ( !(_BYTE)v2 )
  {
    sub_39367A0(v8);
    return v2;
  }
  v9 = (const char *)sub_3936860(v8, 1);
  v33 = v35;
  if ( !v9 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v10 = strlen(v9);
  v36[0] = v10;
  v11 = v10;
  if ( v10 > 0xF )
  {
    na = v10;
    v26 = (_QWORD *)sub_22409D0((__int64)&v33, v36, 0);
    v11 = na;
    v33 = v26;
    v27 = v26;
    v35[0] = v36[0];
    goto LABEL_19;
  }
  if ( v10 != 1 )
  {
    if ( !v10 )
    {
      v12 = v35;
      goto LABEL_9;
    }
    v27 = v35;
LABEL_19:
    memcpy(v27, v9, v11);
    v10 = v36[0];
    v12 = v33;
    goto LABEL_9;
  }
  LOBYTE(v35[0]) = *v9;
  v12 = v35;
LABEL_9:
  v34 = v10;
  *((_BYTE *)v12 + v10) = 0;
  sub_39367A0(v8);
  v13 = sub_C94910(a1[119] + 539408, v33, v34);
  v14 = sub_33F8870((_QWORD *)a1[8], v13, 7u, 0, 0);
  v16 = *(_QWORD *)(a2 + 80);
  v17 = (_QWORD *)a1[8];
  v18 = *(_QWORD *)(a2 + 48);
  v36[0] = v14;
  v19 = *(_QWORD *)(a2 + 40);
  v20 = *(unsigned int *)(a2 + 68);
  v36[1] = v21;
  v37 = _mm_loadu_si128((const __m128i *)(v19 + 40));
  v31 = v16;
  v38 = _mm_loadu_si128((const __m128i *)v19);
  if ( v16 )
  {
    n = v18;
    v30 = v20;
    sub_B96E90((__int64)&v31, v16, 1);
    v18 = n;
    v20 = v30;
  }
  v32 = *(_DWORD *)(a2 + 72);
  v22 = sub_33E66D0(v17, 1286, (__int64)&v31, v18, v20, v15, v36, 3);
  sub_34158F0(a1[8], a2, v22, v23, v24, v25);
  sub_3421DB0(v22);
  sub_33ECEA0((const __m128i *)a1[8], a2);
  if ( v31 )
    sub_B91220((__int64)&v31, v31);
  if ( v33 != v35 )
    j_j___libc_free_0((unsigned __int64)v33);
  return v2;
}
