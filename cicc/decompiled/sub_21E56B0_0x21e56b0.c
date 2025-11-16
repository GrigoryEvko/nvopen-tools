// Function: sub_21E56B0
// Address: 0x21e56b0
//
void __fastcall sub_21E56B0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        __m128i a8,
        __m128i a9)
{
  __int16 v9; // ax
  __int64 v11; // rbx
  _QWORD *v12; // r13
  __int64 v13; // rsi
  __int64 v14; // r12
  __int64 v15; // r14
  __int64 v16; // r12
  const char *v17; // r14
  char *v18; // rax
  size_t v19; // r10
  char *v20; // rdx
  const char *v21; // r14
  _QWORD *v22; // rax
  size_t v23; // r8
  int v24; // r9d
  __int64 *v25; // r12
  size_t v26; // rax
  char *v27; // r10
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rcx
  __int64 v31; // rax
  __int64 v32; // rdx
  int v33; // r8d
  __int64 v34; // r9
  __int64 v35; // rax
  __int64 v36; // rax
  char *v37; // rax
  char *v38; // rdi
  size_t n; // [rsp+8h] [rbp-A8h]
  size_t na; // [rsp+8h] [rbp-A8h]
  char *v41; // [rsp+10h] [rbp-A0h]
  __int64 v42; // [rsp+20h] [rbp-90h] BYREF
  int v43; // [rsp+28h] [rbp-88h]
  char *s[2]; // [rsp+30h] [rbp-80h] BYREF
  _QWORD v45[2]; // [rsp+40h] [rbp-70h] BYREF
  __int64 v46[2]; // [rsp+50h] [rbp-60h] BYREF
  __m128i v47; // [rsp+60h] [rbp-50h]
  __m128i v48; // [rsp+70h] [rbp-40h]

  v9 = *(_WORD *)(a2 + 24);
  if ( v9 == 203 )
  {
    *(_BYTE *)a1 = 1;
    *(_QWORD *)(a1 + 8) = 0;
    return;
  }
  if ( v9 > 203 )
  {
    if ( v9 > 680 )
    {
      if ( (unsigned __int16)(v9 - 681) <= 2u )
        sub_21E4180(a1, a2, a7, *(double *)a8.m128i_i64, a9);
    }
    else if ( v9 > 677 )
    {
      sub_21E3D30(a1, a2, a7, *(double *)a8.m128i_i64, a9);
    }
    else if ( v9 == 277 )
    {
      v35 = *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL);
      if ( *(_WORD *)(v35 + 24) == 260 )
        *(_QWORD *)(a1 + 8) = **(_QWORD **)(v35 + 32);
    }
  }
  else
  {
    if ( v9 == 45 )
    {
      sub_21E54B0(a1, a2, a7, *(double *)a8.m128i_i64, a9, a3, a4, a5, a6);
      return;
    }
    if ( v9 > 45 )
    {
      if ( (unsigned __int16)(v9 - 201) > 1u )
        return;
      v11 = *(_QWORD *)(a1 + 16);
      v12 = *(_QWORD **)(a1 - 176);
      *(_BYTE *)a1 = 0;
      v13 = *(_QWORD *)(a2 + 72);
      v14 = *(_QWORD *)(a1 + 8);
      v42 = v13;
      if ( v13 )
        sub_1623A60((__int64)&v42, v13, 2);
      v43 = *(_DWORD *)(a2 + 64);
      if ( !v14 || *(_WORD *)(v14 + 24) != 34 )
        goto LABEL_25;
      v15 = *(_QWORD *)(v14 + 88);
      v16 = sub_3936750();
      if ( !(unsigned __int8)sub_39371E0(v15, v16) )
      {
        sub_39367A0(v16);
LABEL_25:
        if ( v42 )
          sub_161E7C0((__int64)&v42, v42);
        return;
      }
      v17 = (const char *)sub_3936860(v16, 1);
      s[0] = (char *)v45;
      if ( !v17 )
        goto LABEL_52;
      v18 = (char *)strlen(v17);
      v46[0] = (__int64)v18;
      v19 = (size_t)v18;
      if ( (unsigned __int64)v18 > 0xF )
      {
        na = (size_t)v18;
        v37 = (char *)sub_22409D0(s, v46, 0);
        v19 = na;
        s[0] = v37;
        v38 = v37;
        v45[0] = v46[0];
      }
      else
      {
        if ( v18 == (char *)1 )
        {
          LOBYTE(v45[0]) = *v17;
          v20 = (char *)v45;
          goto LABEL_15;
        }
        if ( !v18 )
        {
          v20 = (char *)v45;
          goto LABEL_15;
        }
        v38 = (char *)v45;
      }
      memcpy(v38, v17, v19);
      v18 = (char *)v46[0];
      v20 = s[0];
LABEL_15:
      s[1] = v18;
      v18[(_QWORD)v20] = 0;
      sub_39367A0(v16);
      v21 = s[0];
      v22 = (_QWORD *)sub_22077B0(32);
      v25 = v22;
      if ( !v22 )
        goto LABEL_21;
      *v22 = v22 + 2;
      v41 = (char *)(v22 + 2);
      if ( v21 )
      {
        v26 = strlen(v21);
        v27 = v41;
        v46[0] = v26;
        v23 = v26;
        if ( v26 > 0xF )
        {
          n = v26;
          v36 = sub_22409D0(v25, v46, 0);
          v23 = n;
          *v25 = v36;
          v27 = (char *)v36;
          v25[2] = v46[0];
        }
        else
        {
          if ( v26 == 1 )
          {
            *((_BYTE *)v25 + 16) = *v21;
LABEL_20:
            v25[1] = v26;
            v27[v26] = 0;
LABEL_21:
            v28 = *(unsigned int *)(v11 + 83296);
            if ( (unsigned int)v28 >= *(_DWORD *)(v11 + 83300) )
            {
              sub_16CD150(v11 + 83288, (const void *)(v11 + 83304), 0, 8, v23, v24);
              v28 = *(unsigned int *)(v11 + 83296);
            }
            *(_QWORD *)(*(_QWORD *)(v11 + 83288) + 8 * v28) = v25;
            ++*(_DWORD *)(v11 + 83296);
            v29 = sub_1D2F9D0(v12, (const char *)*v25, 5u, 0, 0);
            v30 = *(_QWORD *)(a2 + 40);
            v46[0] = v29;
            v31 = *(_QWORD *)(a2 + 32);
            v46[1] = v32;
            v47 = _mm_loadu_si128((const __m128i *)(v31 + 40));
            v33 = *(_DWORD *)(a2 + 60);
            v48 = _mm_loadu_si128((const __m128i *)v31);
            sub_1D23DE0(v12, 380, (__int64)&v42, v30, v33, v34, v46, 3);
            if ( (_QWORD *)s[0] != v45 )
              j_j___libc_free_0(s[0], v45[0] + 1LL);
            goto LABEL_25;
          }
          if ( !v26 )
            goto LABEL_20;
        }
        memcpy(v27, v21, v23);
        v26 = v46[0];
        v27 = (char *)*v25;
        goto LABEL_20;
      }
LABEL_52:
      sub_426248((__int64)"basic_string::_M_construct null not valid");
    }
    if ( v9 == 43 )
    {
      sub_21E39F0(a1, a2, a7, *(double *)a8.m128i_i64, a9, a3, a4, a5, a6);
    }
    else if ( v9 == 44 )
    {
      sub_21E4A10(a1, a2, a7, a8, a9, a3, a4, a5, a6);
    }
  }
}
