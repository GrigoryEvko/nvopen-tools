// Function: sub_2AFB210
// Address: 0x2afb210
//
__int64 *__fastcall sub_2AFB210(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 *a5, int a6)
{
  __int64 *result; // rax
  __int64 v8; // rax
  __int64 v9; // r15
  unsigned __int64 v10; // rax
  char v11; // al
  __int64 v12; // rdx
  __int64 v13; // r8
  char v14; // al
  __int64 v15; // rdx
  __int64 v16; // rdi
  unsigned int *v17; // r10
  __int64 v18; // r11
  __int64 v19; // rbx
  __int64 (__fastcall *v20)(__int64, __int64, unsigned int *, __int64, __int64, __int64); // rax
  __int64 v21; // rax
  _QWORD *v22; // r12
  _QWORD *v23; // rax
  __int64 v24; // r9
  __int64 v25; // rbx
  unsigned int *v26; // r14
  __int64 v27; // r12
  __int64 v28; // rdx
  unsigned int v29; // esi
  __int64 v30; // r13
  unsigned __int64 *v31; // r12
  unsigned __int64 *v32; // rbx
  unsigned __int64 v33; // rdi
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rax
  unsigned int *v37; // r14
  __int64 v38; // rbx
  __int64 v39; // rdx
  unsigned int v40; // esi
  __int64 v41; // rax
  __int64 v42; // [rsp-10h] [rbp-140h]
  __int64 v44; // [rsp+8h] [rbp-128h]
  unsigned int *v45; // [rsp+18h] [rbp-118h]
  unsigned int *v46; // [rsp+18h] [rbp-118h]
  __int64 v47; // [rsp+20h] [rbp-110h]
  unsigned int *v48; // [rsp+20h] [rbp-110h]
  __int64 v49; // [rsp+20h] [rbp-110h]
  __int64 v50; // [rsp+28h] [rbp-108h]
  __int64 v51; // [rsp+30h] [rbp-100h]
  __int64 *v52; // [rsp+38h] [rbp-F8h]
  __int64 v55; // [rsp+58h] [rbp-D8h]
  char v57; // [rsp+6Fh] [rbp-C1h]
  _QWORD v58[4]; // [rsp+70h] [rbp-C0h] BYREF
  __int16 v59; // [rsp+90h] [rbp-A0h]
  _QWORD v60[4]; // [rsp+A0h] [rbp-90h] BYREF
  __int16 v61; // [rsp+C0h] [rbp-70h]
  _BYTE v62[32]; // [rsp+D0h] [rbp-60h] BYREF
  __int16 v63; // [rsp+F0h] [rbp-40h]

  result = (__int64 *)a5[1];
  v52 = result;
  if ( result != (__int64 *)*a5 )
  {
    v8 = 0x2E8BA2E8BA2E8BA3LL * (((__int64)result - *a5) >> 3);
    if ( a6 == 1 )
    {
      if ( v8 != *(_QWORD *)(a1 + 168) )
      {
LABEL_4:
        v9 = *a5;
        while ( 1 )
        {
          v10 = *(_QWORD *)(v9 + 80);
          v57 = -1;
          if ( v10 )
          {
            _BitScanReverse64(&v10, v10);
            v57 = 63 - (v10 ^ 0x3F);
          }
          v11 = *(_BYTE *)(a4 + 32);
          if ( v11 )
          {
            if ( v11 == 1 )
            {
              v58[0] = ".gep";
              v59 = 259;
            }
            else
            {
              if ( *(_BYTE *)(a4 + 33) == 1 )
              {
                v12 = *(_QWORD *)a4;
                v51 = *(_QWORD *)(a4 + 8);
              }
              else
              {
                v12 = a4;
                v11 = 2;
              }
              v58[0] = v12;
              LOBYTE(v59) = v11;
              v58[1] = v51;
              v58[2] = ".gep";
              HIBYTE(v59) = 3;
            }
          }
          else
          {
            v59 = 256;
          }
          v55 = sub_921130(
                  (unsigned int **)a2,
                  *(_QWORD *)(a1 + 96),
                  *(_QWORD *)(a1 + 88),
                  *(_BYTE ***)(v9 + 32),
                  *(unsigned int *)(v9 + 40),
                  (__int64)v58,
                  3u);
          v14 = *(_BYTE *)(a4 + 32);
          if ( v14 )
          {
            if ( v14 == 1 )
            {
              v60[0] = ".extract";
              v61 = 259;
            }
            else
            {
              if ( *(_BYTE *)(a4 + 33) == 1 )
              {
                v15 = *(_QWORD *)a4;
                v50 = *(_QWORD *)(a4 + 8);
              }
              else
              {
                v15 = a4;
                v14 = 2;
              }
              v60[0] = v15;
              LOBYTE(v61) = v14;
              v60[1] = v50;
              v60[2] = ".extract";
              HIBYTE(v61) = 3;
            }
          }
          else
          {
            v13 = 256;
            v61 = 256;
          }
          v16 = *(_QWORD *)(a2 + 80);
          v17 = *(unsigned int **)v9;
          v18 = *(unsigned int *)(v9 + 8);
          v19 = *a3;
          v20 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int *, __int64, __int64, __int64))(*(_QWORD *)v16 + 80LL);
          if ( (char *)v20 == (char *)sub_92FAE0 )
          {
            if ( *(_BYTE *)v19 > 0x15u )
              goto LABEL_41;
            v45 = *(unsigned int **)v9;
            v47 = *(unsigned int *)(v9 + 8);
            v21 = sub_AAADB0(*a3, *(unsigned int **)v9, v47);
            v18 = v47;
            v17 = v45;
            v22 = (_QWORD *)v21;
          }
          else
          {
            v46 = *(unsigned int **)v9;
            v49 = *(unsigned int *)(v9 + 8);
            v41 = v20(v16, v19, v17, v18, v13, v42);
            v17 = v46;
            v18 = v49;
            v22 = (_QWORD *)v41;
          }
          if ( !v22 )
          {
LABEL_41:
            v44 = v18;
            v63 = 257;
            v48 = v17;
            v22 = sub_BD2C40(104, 1u);
            if ( v22 )
            {
              v34 = sub_B501B0(*(_QWORD *)(v19 + 8), v48, v44);
              sub_B44260((__int64)v22, v34, 64, 1u, 0, 0);
              if ( *(v22 - 4) )
              {
                v35 = *(v22 - 3);
                *(_QWORD *)*(v22 - 2) = v35;
                if ( v35 )
                  *(_QWORD *)(v35 + 16) = *(v22 - 2);
              }
              *(v22 - 4) = v19;
              v36 = *(_QWORD *)(v19 + 16);
              *(v22 - 3) = v36;
              if ( v36 )
                *(_QWORD *)(v36 + 16) = v22 - 3;
              *(v22 - 2) = v19 + 16;
              *(_QWORD *)(v19 + 16) = v22 - 4;
              v22[9] = v22 + 11;
              v22[10] = 0x400000000LL;
              sub_B50030((__int64)v22, v48, v44, (__int64)v62);
            }
            (*(void (__fastcall **)(_QWORD, _QWORD *, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
              *(_QWORD *)(a2 + 88),
              v22,
              v60,
              *(_QWORD *)(a2 + 56),
              *(_QWORD *)(a2 + 64));
            v37 = *(unsigned int **)a2;
            v38 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
            if ( *(_QWORD *)a2 != v38 )
            {
              do
              {
                v39 = *((_QWORD *)v37 + 1);
                v40 = *v37;
                v37 += 4;
                sub_B99FD0((__int64)v22, v40, v39);
              }
              while ( (unsigned int *)v38 != v37 );
            }
          }
          v63 = 257;
          v23 = sub_BD2C40(80, unk_3F10A10);
          v25 = (__int64)v23;
          if ( v23 )
            sub_B4D3C0((__int64)v23, (__int64)v22, v55, 0, v57, v24, 0, 0);
          (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
            *(_QWORD *)(a2 + 88),
            v25,
            v62,
            *(_QWORD *)(a2 + 56),
            *(_QWORD *)(a2 + 64));
          v26 = *(unsigned int **)a2;
          v27 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
          if ( *(_QWORD *)a2 != v27 )
          {
            do
            {
              v28 = *((_QWORD *)v26 + 1);
              v29 = *v26;
              v26 += 4;
              sub_B99FD0(v25, v29, v28);
            }
            while ( (unsigned int *)v27 != v26 );
          }
          v9 += 88;
          if ( v52 == (__int64 *)v9 )
            goto LABEL_26;
        }
      }
    }
    else if ( v8 != *(_QWORD *)(a1 + 160) )
    {
      goto LABEL_4;
    }
    sub_2AFA890((__int64 *)a1, a2, a3, (__int64 *)a4, a5, a6);
LABEL_26:
    if ( *(_BYTE *)(a1 + 184) )
      *(_BYTE *)(a1 + 184) = 0;
    result = a5;
    v30 = *a5;
    v31 = (unsigned __int64 *)a5[1];
    if ( (unsigned __int64 *)*a5 != v31 )
    {
      v32 = (unsigned __int64 *)*a5;
      do
      {
        v33 = v32[4];
        if ( (unsigned __int64 *)v33 != v32 + 6 )
          _libc_free(v33);
        if ( (unsigned __int64 *)*v32 != v32 + 2 )
          _libc_free(*v32);
        v32 += 11;
      }
      while ( v31 != v32 );
      result = a5;
      a5[1] = v30;
    }
  }
  return result;
}
