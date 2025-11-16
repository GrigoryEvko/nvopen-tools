// Function: sub_D9A5D0
// Address: 0xd9a5d0
//
__int64 __fastcall sub_D9A5D0(__int64 a1, _BYTE *a2, __int64 a3)
{
  _BYTE *v3; // rbx
  __int64 v4; // rax
  bool v5; // zf
  __int64 v6; // r8
  __int64 v7; // r9
  unsigned __int64 v8; // rdx
  __int64 result; // rax
  _BYTE *v10; // r8
  _BYTE *v11; // rcx
  __int64 v12; // rdi
  unsigned __int16 v13; // dx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 *v19; // r12
  __int64 *v20; // r15
  __int64 v21; // rbx
  _QWORD *v22; // rax
  __int64 v23; // rdi
  _QWORD *v24; // rax
  __int64 v25; // rax
  unsigned __int64 v26; // rdx
  __int64 v27; // [rsp+8h] [rbp-F8h] BYREF
  __int64 *v28; // [rsp+10h] [rbp-F0h]
  _BYTE *v29; // [rsp+18h] [rbp-E8h] BYREF
  __int64 v30; // [rsp+20h] [rbp-E0h]
  _BYTE v31[64]; // [rsp+28h] [rbp-D8h] BYREF
  __int64 v32; // [rsp+68h] [rbp-98h] BYREF
  _QWORD *v33; // [rsp+70h] [rbp-90h]
  __int64 v34; // [rsp+78h] [rbp-88h]
  int v35; // [rsp+80h] [rbp-80h]
  char v36; // [rsp+84h] [rbp-7Ch]
  _BYTE *v37; // [rsp+88h] [rbp-78h] BYREF

  v3 = a2;
  v28 = &v27;
  v30 = 0x800000000LL;
  v33 = &v37;
  v34 = 0x100000008LL;
  v4 = 0;
  v5 = *((_WORD *)a2 + 12) == 8;
  v27 = a3;
  v29 = v31;
  v35 = 0;
  v36 = 1;
  v37 = a2;
  v32 = 1;
  if ( v5 )
  {
    a2 = (_BYTE *)*((_QWORD *)a2 + 6);
    sub_AE6EC0(a3, (__int64)a2);
    v4 = (unsigned int)v30;
    v8 = (unsigned int)v30 + 1LL;
    if ( HIDWORD(v30) < v8 )
    {
      a2 = v31;
      sub_C8D5F0((__int64)&v29, v31, v8, 8u, v6, v7);
      v4 = (unsigned int)v30;
    }
  }
  *(_QWORD *)&v29[8 * v4] = v3;
  result = (unsigned int)(v30 + 1);
  LODWORD(v30) = v30 + 1;
  while ( 1 )
  {
    v10 = v29;
    v11 = &v29[8 * (unsigned int)result];
    if ( !(_DWORD)result )
      break;
    while ( 1 )
    {
      v12 = *((_QWORD *)v11 - 1);
      result = (unsigned int)(result - 1);
      LODWORD(v30) = result;
      v13 = *(_WORD *)(v12 + 24);
      if ( v13 > 0xEu )
      {
        if ( v13 != 15 )
          BUG();
        goto LABEL_8;
      }
      if ( v13 > 1u )
        break;
LABEL_8:
      v11 -= 8;
      if ( !(_DWORD)result )
        goto LABEL_9;
    }
    v14 = sub_D960E0(v12);
    v19 = (__int64 *)(v14 + 8 * v15);
    v20 = (__int64 *)v14;
    if ( (__int64 *)v14 != v19 )
    {
      while ( 1 )
      {
        v21 = *v20;
        if ( !v36 )
          goto LABEL_24;
        v22 = v33;
        v16 = HIDWORD(v34);
        v15 = (__int64)&v33[HIDWORD(v34)];
        if ( v33 != (_QWORD *)v15 )
        {
          while ( v21 != *v22 )
          {
            if ( (_QWORD *)v15 == ++v22 )
              goto LABEL_35;
          }
          goto LABEL_22;
        }
LABEL_35:
        if ( HIDWORD(v34) < (unsigned int)v34 )
        {
          v16 = (unsigned int)++HIDWORD(v34);
          *(_QWORD *)v15 = v21;
          ++v32;
LABEL_25:
          if ( *(_WORD *)(v21 + 24) == 8 )
          {
            a2 = *(_BYTE **)(v21 + 48);
            v23 = *v28;
            if ( !*(_BYTE *)(*v28 + 28) )
              goto LABEL_38;
            v24 = *(_QWORD **)(v23 + 8);
            v16 = *(unsigned int *)(v23 + 20);
            v15 = (__int64)&v24[v16];
            if ( v24 != (_QWORD *)v15 )
            {
              while ( a2 != (_BYTE *)*v24 )
              {
                if ( (_QWORD *)v15 == ++v24 )
                  goto LABEL_37;
              }
              goto LABEL_31;
            }
LABEL_37:
            if ( (unsigned int)v16 < *(_DWORD *)(v23 + 16) )
            {
              *(_DWORD *)(v23 + 20) = v16 + 1;
              *(_QWORD *)v15 = a2;
              ++*(_QWORD *)v23;
            }
            else
            {
LABEL_38:
              sub_C8CC70(v23, (__int64)a2, v15, v16, v17, v18);
            }
          }
LABEL_31:
          v25 = (unsigned int)v30;
          v16 = HIDWORD(v30);
          v26 = (unsigned int)v30 + 1LL;
          if ( v26 > HIDWORD(v30) )
          {
            a2 = v31;
            sub_C8D5F0((__int64)&v29, v31, v26, 8u, v17, v18);
            v25 = (unsigned int)v30;
          }
          v15 = (__int64)v29;
          ++v20;
          *(_QWORD *)&v29[8 * v25] = v21;
          LODWORD(v30) = v30 + 1;
          if ( v19 == v20 )
            break;
        }
        else
        {
LABEL_24:
          a2 = (_BYTE *)*v20;
          sub_C8CC70((__int64)&v32, *v20, v15, v16, v17, v18);
          if ( (_BYTE)v15 )
            goto LABEL_25;
LABEL_22:
          if ( v19 == ++v20 )
            break;
        }
      }
    }
    result = (unsigned int)v30;
  }
LABEL_9:
  if ( !v36 )
  {
    result = _libc_free(v33, a2);
    v10 = v29;
  }
  if ( v10 != v31 )
    return _libc_free(v10, a2);
  return result;
}
