// Function: sub_108C0B0
// Address: 0x108c0b0
//
void *__fastcall sub_108C0B0(__int64 a1, const void *a2, size_t a3, int a4, char a5, _QWORD *a6)
{
  void *v7; // r14
  __int64 v10; // rsi
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rsi
  char *v14; // r8
  __int64 v15; // rdi
  __int64 v16; // rcx
  __int64 v17; // rdx
  char **v18; // rbx
  char **v19; // r13
  char *v20; // rsi
  __int64 v21; // r10
  __int64 v22; // rdx
  char *v24; // [rsp+0h] [rbp-90h]
  __int64 v25; // [rsp+8h] [rbp-88h]
  char *v28; // [rsp+20h] [rbp-70h] BYREF
  __int64 v29; // [rsp+28h] [rbp-68h]
  __int64 v30; // [rsp+30h] [rbp-60h]
  __int64 v31; // [rsp+38h] [rbp-58h]
  char *v32; // [rsp+40h] [rbp-50h] BYREF
  __int64 v33; // [rsp+48h] [rbp-48h]
  __int64 v34; // [rsp+50h] [rbp-40h]
  __int64 v35; // [rsp+58h] [rbp-38h]

  v7 = (void *)(a1 + 8);
  *(_QWORD *)a1 = off_497C020;
  *(_DWORD *)(a1 + 52) = a4;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_DWORD *)(a1 + 48) = 0;
  *(_WORD *)(a1 + 56) = -3;
  memcpy((void *)(a1 + 8), a2, a3);
  v10 = a6[6] - a6[7];
  *(_QWORD *)a1 = off_497C050;
  v11 = a6[9] - a6[5];
  *(_BYTE *)(a1 + 58) = a5;
  v12 = (v10 >> 3) + (((v11 >> 3) - 1) << 6);
  v13 = a6[4] - a6[2];
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 136) = 0;
  sub_108B4A0((__int64 *)(a1 + 64), v12 + (v13 >> 3));
  v14 = *(char **)(a1 + 80);
  v15 = *(_QWORD *)(a1 + 88);
  v16 = *(_QWORD *)(a1 + 96);
  v17 = *(_QWORD *)(a1 + 104);
  v25 = a6[6];
  v18 = (char **)a6[9];
  v19 = (char **)a6[5];
  v20 = (char *)a6[2];
  v24 = (char *)a6[7];
  v21 = a6[4];
  if ( v18 == v19 )
  {
    v35 = v17;
    v33 = v15;
    v34 = v16;
    v32 = v14;
    sub_108B250(&v28, v20, v25, &v32);
  }
  else
  {
    v30 = v16;
    v31 = v17;
    v22 = v21;
    v28 = v14;
    v29 = v15;
    while ( 1 )
    {
      ++v19;
      sub_108B250(&v32, v20, v22, &v28);
      if ( v18 == v19 )
        break;
      v30 = v34;
      v31 = v35;
      v28 = v32;
      v29 = v33;
      v20 = *v19;
      v22 = (__int64)(*v19 + 512);
    }
    sub_108B250(&v28, v24, v25, &v32);
  }
  return memcpy(v7, a2, a3);
}
