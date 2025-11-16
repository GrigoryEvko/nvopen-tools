// Function: sub_1493080
// Address: 0x1493080
//
__int64 *__fastcall sub_1493080(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __m128i a5, __m128i a6)
{
  __int64 *v7; // r12
  unsigned __int64 v8; // rdi
  _BYTE *v9; // rdx
  _BYTE *v10; // r13
  _QWORD *v11; // rax
  __int64 v12; // rsi
  _QWORD *v13; // rbx
  _QWORD *v15; // r8
  _QWORD *v16; // r9
  _QWORD *v17; // rax
  _QWORD *v18; // rdi
  unsigned int v19; // r10d
  _QWORD *v20; // rax
  _QWORD *v21; // rcx
  __int64 v22[2]; // [rsp+0h] [rbp-B0h] BYREF
  __int64 v23; // [rsp+10h] [rbp-A0h]
  __int64 v24; // [rsp+18h] [rbp-98h]
  int v25; // [rsp+20h] [rbp-90h]
  __int64 *v26; // [rsp+28h] [rbp-88h]
  __int64 v27; // [rsp+30h] [rbp-80h]
  __int64 v28; // [rsp+38h] [rbp-78h]
  __int64 v29; // [rsp+40h] [rbp-70h] BYREF
  _BYTE *v30; // [rsp+48h] [rbp-68h]
  _BYTE *v31; // [rsp+50h] [rbp-60h]
  __int64 v32; // [rsp+58h] [rbp-58h]
  int v33; // [rsp+60h] [rbp-50h]
  _BYTE v34[72]; // [rsp+68h] [rbp-48h] BYREF

  v22[0] = a1;
  v28 = a3;
  v26 = &v29;
  v29 = 0;
  v30 = v34;
  v31 = v34;
  v32 = 4;
  v33 = 0;
  v22[1] = 0;
  v23 = 0;
  v24 = 0;
  v25 = 0;
  v27 = 0;
  v7 = sub_14927C0(v22, a2, a5, a6);
  j___libc_free_0(v23);
  v8 = (unsigned __int64)v31;
  v9 = v30;
  if ( *((_WORD *)v7 + 12) == 7 )
  {
    if ( v31 == v30 )
      v10 = &v31[8 * HIDWORD(v32)];
    else
      v10 = &v31[8 * (unsigned int)v32];
    if ( v31 != v10 )
    {
      v11 = v31;
      while ( 1 )
      {
        v12 = *v11;
        v13 = v11;
        if ( *v11 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v10 == (_BYTE *)++v11 )
          goto LABEL_8;
      }
      if ( v11 != (_QWORD *)v10 )
      {
        v15 = *(_QWORD **)(a4 + 16);
        v16 = *(_QWORD **)(a4 + 8);
        if ( v15 == v16 )
          goto LABEL_20;
LABEL_13:
        sub_16CCBA0(a4, v12);
        v15 = *(_QWORD **)(a4 + 16);
        v16 = *(_QWORD **)(a4 + 8);
LABEL_14:
        while ( 1 )
        {
          v17 = v13 + 1;
          if ( v13 + 1 == (_QWORD *)v10 )
            break;
          while ( 1 )
          {
            v12 = *v17;
            v13 = v17;
            if ( *v17 < 0xFFFFFFFFFFFFFFFELL )
              break;
            if ( v10 == (_BYTE *)++v17 )
              goto LABEL_17;
          }
          if ( v17 == (_QWORD *)v10 )
            break;
          if ( v15 != v16 )
            goto LABEL_13;
LABEL_20:
          v18 = &v15[*(unsigned int *)(a4 + 28)];
          v19 = *(_DWORD *)(a4 + 28);
          if ( v18 == v15 )
          {
LABEL_29:
            if ( v19 >= *(_DWORD *)(a4 + 24) )
              goto LABEL_13;
            *(_DWORD *)(a4 + 28) = v19 + 1;
            *v18 = v12;
            v16 = *(_QWORD **)(a4 + 8);
            ++*(_QWORD *)a4;
            v15 = *(_QWORD **)(a4 + 16);
          }
          else
          {
            v20 = v15;
            v21 = 0;
            while ( *v20 != v12 )
            {
              if ( *v20 == -2 )
                v21 = v20;
              if ( v18 == ++v20 )
              {
                if ( !v21 )
                  goto LABEL_29;
                *v21 = v12;
                v15 = *(_QWORD **)(a4 + 16);
                --*(_DWORD *)(a4 + 32);
                v16 = *(_QWORD **)(a4 + 8);
                ++*(_QWORD *)a4;
                goto LABEL_14;
              }
            }
          }
        }
LABEL_17:
        v8 = (unsigned __int64)v31;
        v9 = v30;
      }
    }
  }
  else
  {
    v7 = 0;
  }
LABEL_8:
  if ( v9 != (_BYTE *)v8 )
    _libc_free(v8);
  return v7;
}
