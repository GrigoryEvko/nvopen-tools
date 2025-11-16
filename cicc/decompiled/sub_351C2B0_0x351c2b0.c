// Function: sub_351C2B0
// Address: 0x351c2b0
//
__int64 *__fastcall sub_351C2B0(
        __int64 *a1,
        __int64 *a2,
        char *a3,
        __int64 a4,
        __int64 a5,
        __int64 *a6,
        __int64 a7,
        __int64 a8)
{
  __int64 v8; // rax
  __int64 *v9; // r15
  __int64 *v10; // r12
  __int64 *v11; // rbx
  __int64 v12; // r14
  __int64 v13; // r13
  __int64 v14; // r12
  __int64 *v15; // r11
  __int64 *v16; // r10
  __int64 *v17; // rax
  int v18; // r11d
  __int64 v19; // r8
  __int64 *v20; // r10
  __int64 *v21; // r15
  __int64 v22; // r14
  char *v23; // rax
  void *v24; // r8
  __int64 v25; // rdx
  signed __int64 v26; // r13
  __int64 *result; // rax
  __int64 v28; // rax
  __int64 v29; // r13
  __int64 v30; // rax
  __int64 *v31; // rsi
  __int64 *v32; // rdi
  size_t v33; // rdx
  __int64 *v34; // r13
  __int64 *v35; // rbx
  __int64 v36; // r15
  __int64 *i; // r14
  __int64 v38; // r13
  unsigned __int64 v39; // r12
  __int64 *v40; // rax
  __int64 *v41; // r15
  int src; // [rsp+8h] [rbp-68h]
  __int64 *v43; // [rsp+10h] [rbp-60h]
  void *v44; // [rsp+10h] [rbp-60h]
  int v45; // [rsp+10h] [rbp-60h]
  __int64 *v46; // [rsp+18h] [rbp-58h]
  __int64 v47; // [rsp+18h] [rbp-58h]
  char *v48; // [rsp+18h] [rbp-58h]
  __int64 *v49; // [rsp+18h] [rbp-58h]
  __int64 *v50; // [rsp+20h] [rbp-50h]
  char *v51; // [rsp+28h] [rbp-48h]
  __int64 *v52; // [rsp+28h] [rbp-48h]
  void *destb; // [rsp+30h] [rbp-40h]
  __int64 *desta; // [rsp+30h] [rbp-40h]

  v8 = a5;
  v9 = a2;
  v10 = a1;
  v11 = a6;
  v12 = a8;
  if ( a7 <= a5 )
    v8 = a7;
  if ( a4 <= v8 )
  {
LABEL_12:
    v26 = (char *)v9 - (char *)v10;
    if ( v10 != v9 )
      memmove(v11, v10, (char *)v9 - (char *)v10);
    result = (__int64 *)((char *)v11 + v26);
    v51 = (char *)v11 + v26;
    if ( v11 != (__int64 *)((char *)v11 + v26) )
    {
      while ( a3 != (char *)v9 )
      {
        v29 = *v9;
        destb = (void *)sub_2F06CB0(*(_QWORD *)(v12 + 536), *v11);
        if ( (unsigned __int64)destb < sub_2F06CB0(*(_QWORD *)(v12 + 536), v29) )
        {
          v28 = *v9;
          ++v10;
          ++v9;
          *(v10 - 1) = v28;
          if ( v51 == (char *)v11 )
            break;
        }
        else
        {
          v30 = *v11;
          ++v10;
          ++v11;
          *(v10 - 1) = v30;
          if ( v51 == (char *)v11 )
            break;
        }
      }
      result = (__int64 *)v51;
      if ( v11 != (__int64 *)v51 )
      {
        v31 = v11;
        v32 = v10;
        v33 = v51 - (char *)v11;
        return (__int64 *)memmove(v32, v31, v33);
      }
    }
  }
  else
  {
    v13 = a5;
    if ( a7 < a5 )
    {
      v14 = a4;
      v15 = a1;
      v16 = a2;
      while ( 1 )
      {
        if ( v13 < v14 )
        {
          v45 = (int)v15;
          v49 = v16;
          v22 = v14 / 2;
          v21 = &v15[v14 / 2];
          v40 = sub_35116C0(v16, (__int64)a3, v21, a8);
          v20 = v49;
          v18 = v45;
          v50 = v40;
          v19 = v40 - v49;
        }
        else
        {
          v43 = v16;
          v46 = v15;
          v50 = &v16[v13 / 2];
          v17 = sub_3511610(v15, (__int64)v16, v50, a8);
          v18 = (int)v46;
          v19 = v13 / 2;
          v20 = v43;
          v21 = v17;
          v22 = v17 - v46;
        }
        v14 -= v22;
        src = v18;
        v47 = v19;
        v23 = sub_351BCC0(v21, v20, v50, v14, v19, a6, a7);
        v24 = (void *)v47;
        v48 = v23;
        v44 = v24;
        sub_351C2B0(src, (_DWORD)v21, (_DWORD)v23, v22, (_DWORD)v24, (_DWORD)a6, a7, a8);
        v25 = a7;
        v13 -= (__int64)v44;
        if ( v13 <= a7 )
          v25 = v13;
        if ( v14 <= v25 )
        {
          v12 = a8;
          v11 = a6;
          v10 = (__int64 *)v48;
          v9 = v50;
          goto LABEL_12;
        }
        if ( v13 <= a7 )
          break;
        v16 = v50;
        v15 = (__int64 *)v48;
      }
      v12 = a8;
      v11 = a6;
      v10 = (__int64 *)v48;
      v9 = v50;
    }
    if ( a3 != (char *)v9 )
      memmove(v11, v9, a3 - (char *)v9);
    result = (__int64 *)((char *)v11 + a3 - (char *)v9);
    if ( v10 == v9 )
    {
      if ( v11 != result )
      {
        v33 = a3 - (char *)v9;
        v32 = v9;
        goto LABEL_39;
      }
    }
    else if ( v11 != result )
    {
      v52 = v10;
      v34 = v9 - 1;
      desta = v11;
      v35 = result - 1;
      v36 = v12;
      for ( i = v34; ; --i )
      {
        while ( 1 )
        {
          v38 = *v35;
          v39 = sub_2F06CB0(*(_QWORD *)(v36 + 536), *i);
          a3 -= 8;
          if ( v39 < sub_2F06CB0(*(_QWORD *)(v36 + 536), v38) )
            break;
          result = (__int64 *)*v35;
          *(_QWORD *)a3 = *v35;
          if ( desta == v35 )
            return result;
          --v35;
        }
        result = (__int64 *)*i;
        *(_QWORD *)a3 = *i;
        if ( i == v52 )
          break;
      }
      v41 = v35;
      v11 = desta;
      if ( desta != v41 + 1 )
      {
        v33 = (char *)(v41 + 1) - (char *)desta;
        v32 = (__int64 *)&a3[-v33];
LABEL_39:
        v31 = v11;
        return (__int64 *)memmove(v32, v31, v33);
      }
    }
  }
  return result;
}
