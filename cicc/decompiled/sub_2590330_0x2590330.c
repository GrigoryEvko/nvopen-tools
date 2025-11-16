// Function: sub_2590330
// Address: 0x2590330
//
__int64 __fastcall sub_2590330(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rbx
  __int64 v9; // rax
  __int64 v10; // rax
  int v11; // r14d
  __int64 v12; // r12
  __int64 v13; // rax
  int v14; // edx
  unsigned int v15; // eax
  __int64 v16; // r8
  unsigned __int8 *v17; // rax
  char v18; // al
  __int64 v19; // rax
  __int64 v20; // rbx
  unsigned __int64 v21; // rdi
  unsigned __int64 v22; // rcx
  __int64 v23; // rax
  __int64 v25; // rbx
  __int64 v26; // r12
  size_t v27; // rbx
  int v28; // r9d
  int v29; // r8d
  unsigned int v30; // eax
  __int64 v31; // rcx
  __int64 v32; // [rsp+8h] [rbp-C8h]
  void *v33; // [rsp+10h] [rbp-C0h]
  __int64 v36; // [rsp+38h] [rbp-98h]
  char *v37; // [rsp+40h] [rbp-90h]
  __int64 v38; // [rsp+40h] [rbp-90h]
  __int64 v39; // [rsp+48h] [rbp-88h]
  char v40; // [rsp+57h] [rbp-79h] BYREF
  __int64 v41; // [rsp+58h] [rbp-78h] BYREF
  __int64 v42; // [rsp+60h] [rbp-70h] BYREF
  void *v43; // [rsp+68h] [rbp-68h]
  __int64 v44; // [rsp+70h] [rbp-60h]
  __int64 v45; // [rsp+78h] [rbp-58h]
  __int64 v46; // [rsp+80h] [rbp-50h]
  __int64 v47; // [rsp+88h] [rbp-48h]
  __int64 v48; // [rsp+90h] [rbp-40h]
  __int64 v49; // [rsp+98h] [rbp-38h]

  v8 = sub_2568740(a3, a4);
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  sub_C7D6A0(0, 0, 8);
  v9 = *(unsigned int *)(v8 + 24);
  LODWORD(v45) = v9;
  if ( (_DWORD)v9 )
  {
    v43 = (void *)sub_C7D670(8 * v9, 8);
    v44 = *(_QWORD *)(v8 + 16);
    memcpy(v43, *(const void **)(v8 + 8), 8LL * (unsigned int)v45);
  }
  else
  {
    v43 = 0;
    v44 = 0;
  }
  v46 = *(_QWORD *)(v8 + 32);
  v47 = *(_QWORD *)(v8 + 40);
  v48 = *(_QWORD *)(v8 + 48);
  v49 = *(_QWORD *)(v8 + 56);
  sub_C7D6A0(0, 0, 8);
  v10 = *(unsigned int *)(a3 + 224);
  if ( (_DWORD)v10 )
  {
    v27 = 8 * v10;
    v32 = 8 * v10;
    v33 = (void *)sub_C7D670(8 * v10, 8);
    memcpy(v33, *(const void **)(a3 + 208), v27);
  }
  else
  {
    v32 = 0;
    v33 = 0;
  }
  v11 = 0;
  v12 = *(_QWORD *)(a3 + 240);
  v39 = *(_QWORD *)(a3 + 248);
  v36 = *(_QWORD *)(a3 + 256);
  v13 = 0;
  if ( *(_DWORD *)(a5 + 40) )
  {
    while ( 1 )
    {
      v19 = *(_QWORD *)(*(_QWORD *)(a5 + 32) + 8 * v13);
      v20 = *(_QWORD *)(v19 + 24);
      v37 = (char *)v19;
      if ( *(_BYTE *)v20 <= 0x1Cu )
        goto LABEL_9;
      v21 = v20 & 0xFFFFFFFFFFFFFFFBLL;
      v22 = v20 | 4;
      if ( (_DWORD)v45 )
      {
        v14 = v45 - 1;
        v15 = (v45 - 1) & (v22 ^ (v22 >> 9));
        v16 = *((_QWORD *)v43 + v15);
        if ( v22 != v16 )
        {
          v28 = 1;
          while ( v16 != -4 )
          {
            v15 = v14 & (v28 + v15);
            v16 = *((_QWORD *)v43 + v15);
            if ( v22 == v16 )
              goto LABEL_8;
            ++v28;
          }
          v29 = 1;
          v30 = v14 & (v21 ^ (v21 >> 9));
          v31 = *((_QWORD *)v43 + v30);
          if ( v21 != v31 )
          {
            while ( v31 != -4 )
            {
              v30 = v14 & (v29 + v30);
              v31 = *((_QWORD *)v43 + v30);
              if ( v21 == v31 )
                goto LABEL_8;
              ++v29;
            }
            goto LABEL_12;
          }
        }
LABEL_8:
        v40 = 0;
        LOBYTE(v41) = 0;
        v17 = (unsigned __int8 *)sub_250D070((_QWORD *)(a1 + 72));
        sub_258FD00(a2, a1, v17, v37, (unsigned __int8 *)v20, &v40, &v41);
        v18 = v40;
        *(_BYTE *)(a6 + 8) |= v40;
        *(_BYTE *)(a6 + 9) |= v18;
        if ( (_BYTE)v41 )
        {
          v25 = *(_QWORD *)(v20 + 16);
          if ( v25 )
          {
            v38 = v12;
            v26 = v25;
            do
            {
              v41 = v26;
              sub_25789E0(a5, &v41);
              v26 = *(_QWORD *)(v26 + 8);
            }
            while ( v26 );
            v12 = v38;
          }
        }
LABEL_9:
        v13 = (unsigned int)(v11 + 1);
        v11 = v13;
        if ( *(_DWORD *)(a5 + 40) <= (unsigned int)v13 )
          break;
      }
      else
      {
LABEL_12:
        v23 = v47;
        while ( v12 != v23 || v39 != v48 || v36 != v49 )
        {
          v23 = sub_3106C80(&v42);
          v47 = v23;
          if ( v20 == v23 )
            goto LABEL_8;
        }
        v13 = (unsigned int)(v11 + 1);
        v11 = v13;
        if ( *(_DWORD *)(a5 + 40) <= (unsigned int)v13 )
          break;
      }
    }
  }
  sub_C7D6A0((__int64)v33, v32, 8);
  return sub_C7D6A0((__int64)v43, 8LL * (unsigned int)v45, 8);
}
