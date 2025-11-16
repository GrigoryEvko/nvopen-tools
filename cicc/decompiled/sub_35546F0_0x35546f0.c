// Function: sub_35546F0
// Address: 0x35546f0
//
__int64 __fastcall sub_35546F0(__int64 a1, __int64 *a2)
{
  __int64 result; // rax
  __int64 v3; // rbx
  __int64 v4; // rdi
  __int64 v5; // rsi
  __int64 v6; // r14
  _DWORD *v7; // r14
  int v8; // edx
  __int64 v9; // rbx
  _DWORD *v10; // r13
  int v11; // eax
  _BYTE *v12; // r12
  __int64 *v13; // rcx
  __int64 *v14; // r8
  _QWORD *v15; // rdx
  __int64 v16; // rax
  _QWORD *v17; // rsi
  unsigned int v18; // esi
  __int64 v19; // r11
  __int64 v20; // rdi
  __int64 v21; // rsi
  __int64 v22; // rdi
  int v23; // [rsp+8h] [rbp-188h]
  int v24; // [rsp+Ch] [rbp-184h]
  int v25; // [rsp+30h] [rbp-160h]
  __int64 v27; // [rsp+50h] [rbp-140h]
  _QWORD *v28; // [rsp+58h] [rbp-138h]
  __int64 v29; // [rsp+60h] [rbp-130h]
  unsigned int v30; // [rsp+6Ch] [rbp-124h]
  __int64 v31; // [rsp+70h] [rbp-120h]
  __int64 v33; // [rsp+80h] [rbp-110h] BYREF
  __int64 v34; // [rsp+88h] [rbp-108h]
  __int64 v35; // [rsp+90h] [rbp-100h]
  __int64 v36; // [rsp+98h] [rbp-F8h]
  __int64 *v37; // [rsp+A0h] [rbp-F0h]
  __int64 v38; // [rsp+A8h] [rbp-E8h]
  _BYTE v39[64]; // [rsp+B0h] [rbp-E0h] BYREF
  __int64 v40; // [rsp+F0h] [rbp-A0h] BYREF
  __int64 v41; // [rsp+F8h] [rbp-98h]
  __int64 v42; // [rsp+100h] [rbp-90h]
  __int64 v43; // [rsp+108h] [rbp-88h]
  _BYTE *v44; // [rsp+110h] [rbp-80h]
  __int64 v45; // [rsp+118h] [rbp-78h]
  _BYTE v46[112]; // [rsp+120h] [rbp-70h] BYREF

  result = *((unsigned int *)a2 + 2);
  v25 = result;
  if ( (int)result > 0 )
  {
    v3 = 0;
    v30 = result - 2;
    v29 = 0;
    v23 = 0;
    while ( 1 )
    {
      v33 = 0;
      v34 = 0;
      v6 = *a2;
      v37 = (__int64 *)v39;
      v38 = 0x800000000LL;
      v35 = 0;
      v7 = (_DWORD *)(v3 + v6);
      v8 = v7[10];
      v36 = 0;
      if ( !v8 )
      {
        v4 = 0;
        v5 = 0;
LABEL_7:
        sub_C7D6A0(v4, v5, 8);
        v27 = v3 + 88;
        goto LABEL_8;
      }
      if ( !sub_35543E0((__int64)v7, (__int64)&v33, *(_QWORD **)(a1 + 3464), 0) )
      {
        if ( v37 != (__int64 *)v39 )
          _libc_free((unsigned __int64)v37);
        v4 = v34;
        v5 = 8LL * (unsigned int)v36;
        goto LABEL_7;
      }
      v27 = v3 + 88;
      if ( v25 <= (int)v29 + 1 )
        goto LABEL_34;
      v9 = v3 + 88;
      v31 = 88 * (v29 + v30) + 176;
LABEL_14:
      v10 = (_DWORD *)(v9 + *a2);
      if ( v7[13] != v10[13] )
        goto LABEL_13;
      v40 = 0;
      v45 = 0x800000000LL;
      v11 = v10[10];
      v41 = 0;
      v42 = 0;
      v43 = 0;
      v44 = v46;
      if ( !v11 )
        break;
      if ( !sub_35543E0((__int64)v10, (__int64)&v40, *(_QWORD **)(a1 + 3464), 0) )
      {
        if ( v44 != v46 )
          _libc_free((unsigned __int64)v44);
        v20 = v41;
        v21 = 8LL * (unsigned int)v43;
        goto LABEL_44;
      }
      v12 = v44;
      if ( (unsigned int)v38 > (unsigned __int64)(unsigned int)v45 )
        goto LABEL_31;
      v13 = v37;
      v14 = &v37[(unsigned int)v38];
      if ( v37 != v14 )
      {
        v28 = &v44[8 * (unsigned int)v45];
        v15 = &v44[32 * ((unsigned __int64)(unsigned int)v45 >> 2)];
        while ( 1 )
        {
          v16 = *v13;
          if ( !(_DWORD)v42 )
            break;
          if ( !(_DWORD)v43 )
            goto LABEL_31;
          v18 = (v43 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
          v19 = *(_QWORD *)(v41 + 8LL * v18);
          if ( v16 != v19 )
          {
            v24 = 1;
            while ( v19 != -4096 )
            {
              v18 = (v43 - 1) & (v24 + v18);
              ++v24;
              v19 = *(_QWORD *)(v41 + 8LL * v18);
              if ( v16 == v19 )
                goto LABEL_29;
            }
LABEL_31:
            if ( v44 != v46 )
              _libc_free((unsigned __int64)v44);
            v9 += 88;
            sub_C7D6A0(v41, 8LL * (unsigned int)v43, 8);
            if ( v31 == v9 )
              goto LABEL_34;
            goto LABEL_14;
          }
LABEL_29:
          if ( v14 == ++v13 )
            goto LABEL_30;
        }
        if ( (unsigned __int64)(unsigned int)v45 >> 2 )
        {
          v17 = v44;
          while ( v16 != *v17 )
          {
            if ( v16 == v17[1] )
            {
              ++v17;
              break;
            }
            if ( v16 == v17[2] )
            {
              v17 += 2;
              break;
            }
            if ( v16 == v17[3] )
            {
              v17 += 3;
              break;
            }
            v17 += 4;
            if ( v15 == v17 )
            {
              v22 = v28 - v15;
              goto LABEL_52;
            }
          }
LABEL_28:
          if ( v28 == v17 )
            goto LABEL_31;
          goto LABEL_29;
        }
        v22 = (unsigned int)v45;
        v17 = v44;
LABEL_52:
        switch ( v22 )
        {
          case 2LL:
LABEL_56:
            if ( v16 == *v17 )
              goto LABEL_28;
            ++v17;
            break;
          case 3LL:
            if ( v16 == *v17 )
              goto LABEL_28;
            ++v17;
            goto LABEL_56;
          case 1LL:
            break;
          default:
            goto LABEL_31;
        }
        if ( v16 != *v17 )
          goto LABEL_31;
        goto LABEL_28;
      }
LABEL_30:
      if ( (_DWORD)v38 != (_DWORD)v45 )
        goto LABEL_31;
      v7[16] = ++v23;
      v10[16] = v23;
      if ( v12 != v46 )
        _libc_free((unsigned __int64)v12);
      sub_C7D6A0(v41, 8LL * (unsigned int)v43, 8);
LABEL_34:
      if ( v37 != (__int64 *)v39 )
        _libc_free((unsigned __int64)v37);
      sub_C7D6A0(v34, 8LL * (unsigned int)v36, 8);
LABEL_8:
      result = --v30;
      ++v29;
      v3 = v27;
      if ( v30 == -2 )
        return result;
    }
    v20 = 0;
    v21 = 0;
LABEL_44:
    sub_C7D6A0(v20, v21, 8);
LABEL_13:
    v9 += 88;
    if ( v31 == v9 )
      goto LABEL_34;
    goto LABEL_14;
  }
  return result;
}
