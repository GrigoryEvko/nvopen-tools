// Function: sub_1ED0F40
// Address: 0x1ed0f40
//
unsigned __int64 __fastcall sub_1ED0F40(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  unsigned int v4; // r12d
  _DWORD *v5; // rsi
  unsigned int v6; // r8d
  unsigned __int64 result; // rax
  unsigned int v8; // r8d
  int v9; // r9d
  __int64 v10; // r15
  unsigned __int64 v11; // rcx
  __int64 v12; // r13
  __int64 v13; // rdx
  int v14; // r11d
  unsigned int v15; // eax
  __int64 v16; // r10
  __int64 v17; // r14
  __int64 v18; // rsi
  float v19; // xmm0_4
  bool v20; // zf
  unsigned int *v21; // r14
  float *v22; // rdi
  void *v23; // rax
  _QWORD *v24; // r13
  volatile signed __int32 *v25; // rdi
  volatile signed __int32 *v26; // r14
  __int64 v27; // rsi
  unsigned int v28; // eax
  __int64 v29; // rsi
  __int64 v30; // rdx
  _QWORD *v31; // rdi
  _QWORD *v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // [rsp+0h] [rbp-C0h]
  unsigned int v35; // [rsp+8h] [rbp-B8h]
  int v36; // [rsp+Ch] [rbp-B4h]
  __int64 v37; // [rsp+30h] [rbp-90h]
  int v38; // [rsp+38h] [rbp-88h]
  unsigned int v39; // [rsp+3Ch] [rbp-84h]
  float v40; // [rsp+40h] [rbp-80h]
  __int64 v41; // [rsp+40h] [rbp-80h]
  __int64 v42; // [rsp+40h] [rbp-80h]
  __int64 v43; // [rsp+40h] [rbp-80h]
  __int64 v44; // [rsp+48h] [rbp-78h]
  __int64 v45; // [rsp+58h] [rbp-68h] BYREF
  unsigned int v46; // [rsp+60h] [rbp-60h]
  void *dest; // [rsp+68h] [rbp-58h] BYREF
  __int64 v48; // [rsp+70h] [rbp-50h] BYREF
  volatile signed __int32 *v49; // [rsp+78h] [rbp-48h]
  unsigned int v50; // [rsp+80h] [rbp-40h] BYREF
  void *v51; // [rsp+88h] [rbp-38h]

  v3 = *(_QWORD *)(a2 + 8);
  v45 = a2;
  v50 = 0;
  v44 = v3;
  v39 = -1171354717 * ((__int64)(*(_QWORD *)(a2 + 168) - *(_QWORD *)(a2 + 160)) >> 3);
  if ( v39 )
  {
    while ( 1 )
    {
      v5 = *(_DWORD **)(a2 + 192);
      if ( v5 == sub_1ECAFE0(*(_DWORD **)(a2 + 184), (__int64)v5, (int *)&v50) )
        break;
      v4 = v50 + 1;
      v50 = v4;
      if ( v6 <= v4 )
        goto LABEL_6;
    }
    v4 = v50;
  }
  else
  {
    v4 = 0;
  }
LABEL_6:
  result = sub_1ECCC00((__int64)&v45);
  v38 = result;
  if ( (_DWORD)result != v4 )
  {
    v10 = a2;
    v37 = a2 + 88;
    while ( 1 )
    {
      v11 = *(unsigned int *)(v44 + 408);
      v12 = 88LL * v4;
      v13 = v12 + *(_QWORD *)(v10 + 160);
      v14 = *(_DWORD *)(v13 + 40);
      v15 = v14 & 0x7FFFFFFF;
      v16 = v14 & 0x7FFFFFFF;
      v17 = 8 * v16;
      if ( (v14 & 0x7FFFFFFFu) >= (unsigned int)v11 )
        break;
      v18 = *(_QWORD *)(*(_QWORD *)(v44 + 400) + 8LL * v15);
      if ( !v18 )
        break;
      v19 = *(float *)(v18 + 116);
      v20 = v19 == 0.0;
LABEL_11:
      if ( v20 )
        v40 = 1.1754944e-38;
      else
        v40 = v19 + 10.0;
      v21 = *(unsigned int **)v13;
      v46 = **(_DWORD **)v13;
      sub_1ECC890(&dest, v46);
      v22 = (float *)dest;
      if ( 4LL * v46 )
      {
        memmove(dest, *((const void **)v21 + 1), 4LL * v46);
        v22 = (float *)dest;
      }
      *v22 = v40;
      v23 = dest;
      dest = 0;
      v50 = v46;
      v46 = 0;
      v51 = v23;
      sub_1ED0750(&v48, v37, &v50);
      if ( v51 )
        j_j___libc_free_0_0(v51);
      v24 = (_QWORD *)(*(_QWORD *)(v10 + 160) + v12);
      v25 = (volatile signed __int32 *)v24[1];
      *v24 = v48;
      v26 = v49;
      if ( v49 != v25 )
      {
        if ( v49 )
        {
          if ( &_pthread_key_create )
            _InterlockedAdd(v49 + 2, 1u);
          else
            ++*((_DWORD *)v49 + 2);
          v25 = (volatile signed __int32 *)v24[1];
        }
        if ( v25 )
          sub_A191D0(v25);
        v24[1] = v26;
        v25 = v49;
      }
      if ( v25 )
        sub_A191D0(v25);
      if ( dest )
        j_j___libc_free_0_0(dest);
      result = v39;
      v50 = ++v4;
      if ( v39 > v4 )
      {
        while ( 1 )
        {
          v27 = *(_QWORD *)(v10 + 192);
          result = (unsigned __int64)sub_1ECAFE0(*(_DWORD **)(v10 + 184), v27, (int *)&v50);
          if ( v27 == result )
            break;
          result = v50;
          v4 = v50 + 1;
          v50 = v4;
          if ( v8 <= v4 )
            goto LABEL_34;
        }
        v4 = v50;
      }
LABEL_34:
      if ( v38 == v4 )
        return result;
    }
    v28 = v15 + 1;
    if ( (unsigned int)v11 < v28 )
    {
      v30 = v28;
      if ( v28 < v11 )
      {
        *(_DWORD *)(v44 + 408) = v28;
        v29 = *(_QWORD *)(v44 + 400);
        goto LABEL_38;
      }
      if ( v28 > v11 )
      {
        if ( v28 > (unsigned __int64)*(unsigned int *)(v44 + 412) )
        {
          v34 = v14 & 0x7FFFFFFF;
          v35 = v28;
          v36 = v14;
          v43 = v28;
          sub_16CD150(v44 + 400, (const void *)(v44 + 416), v28, 8, v8, v9);
          v16 = v34;
          v14 = v36;
          v30 = v43;
          v11 = *(unsigned int *)(v44 + 408);
          v28 = v35;
        }
        v29 = *(_QWORD *)(v44 + 400);
        v31 = (_QWORD *)(v29 + 8 * v30);
        v32 = (_QWORD *)(v29 + 8 * v11);
        v33 = *(_QWORD *)(v44 + 416);
        if ( v31 != v32 )
        {
          do
            *v32++ = v33;
          while ( v31 != v32 );
          v29 = *(_QWORD *)(v44 + 400);
        }
        *(_DWORD *)(v44 + 408) = v28;
        goto LABEL_38;
      }
    }
    v29 = *(_QWORD *)(v44 + 400);
LABEL_38:
    v41 = v16;
    *(_QWORD *)(v29 + v17) = sub_1DBA290(v14);
    v42 = *(_QWORD *)(*(_QWORD *)(v44 + 400) + 8 * v41);
    sub_1DBB110((_QWORD *)v44, v42);
    v19 = *(float *)(v42 + 116);
    v13 = v12 + *(_QWORD *)(v10 + 160);
    v20 = v19 == 0.0;
    goto LABEL_11;
  }
  return result;
}
