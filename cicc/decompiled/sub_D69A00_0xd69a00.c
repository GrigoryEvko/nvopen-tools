// Function: sub_D69A00
// Address: 0xd69a00
//
__int64 __fastcall sub_D69A00(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        char a6,
        unsigned __int8 (__fastcall *a7)(__int64, _QWORD),
        __int64 a8)
{
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 result; // rax
  unsigned int v12; // ecx
  __int64 *v13; // rdx
  __int64 v14; // r9
  __int64 v15; // r14
  __int64 v16; // rbx
  __int64 v17; // rsi
  int v18; // ecx
  __int64 v19; // rdi
  unsigned int v20; // edx
  __int64 v21; // rax
  __int64 v22; // r8
  _BYTE *v23; // r12
  __int64 v24; // r10
  _BYTE **v25; // rax
  _BYTE *v26; // rax
  int v27; // ecx
  int v28; // eax
  int v29; // edx
  int v30; // r10d
  int v31; // r9d
  __int64 v36; // [rsp+20h] [rbp-70h]
  __int64 v37; // [rsp+28h] [rbp-68h]
  __int64 v38; // [rsp+28h] [rbp-68h]
  unsigned __int64 v39; // [rsp+30h] [rbp-60h] BYREF
  __int64 v40; // [rsp+38h] [rbp-58h] BYREF
  _BYTE *v41; // [rsp+40h] [rbp-50h]
  __int64 v42; // [rsp+48h] [rbp-48h]
  __int64 v43; // [rsp+50h] [rbp-40h]

  v8 = *a1;
  v9 = *(_QWORD *)(*a1 + 72);
  result = *(unsigned int *)(v8 + 88);
  if ( (_DWORD)result )
  {
    v12 = (result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v13 = (__int64 *)(v9 + 16LL * v12);
    v14 = *v13;
    if ( a2 == *v13 )
    {
LABEL_3:
      result = v9 + 16 * result;
      if ( v13 != (__int64 *)result )
      {
        v15 = v13[1];
        if ( v15 )
        {
          v16 = *(_QWORD *)(v15 + 8);
          if ( v16 != v15 )
          {
            while ( 1 )
            {
              if ( !v16 )
                BUG();
              result = (unsigned int)*(unsigned __int8 *)(v16 - 32) - 26;
              if ( (unsigned int)result > 1 )
                goto LABEL_20;
              v27 = *(_DWORD *)(a4 + 24);
              if ( !v27 )
                goto LABEL_24;
              v17 = *(_QWORD *)(v16 + 40);
              v18 = v27 - 1;
              v19 = *(_QWORD *)(a4 + 8);
              v40 = 2;
              v41 = 0;
              v42 = -4096;
              v43 = 0;
              v20 = v18 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
              v21 = v19 + ((unsigned __int64)v20 << 6);
              v22 = *(_QWORD *)(v21 + 24);
              if ( v17 != v22 )
              {
                v28 = 1;
                while ( v22 != -4096 )
                {
                  v31 = v28 + 1;
                  v20 = v18 & (v28 + v20);
                  v21 = v19 + ((unsigned __int64)v20 << 6);
                  v22 = *(_QWORD *)(v21 + 24);
                  if ( v17 == v22 )
                    goto LABEL_8;
                  v28 = v31;
                }
                v39 = (unsigned __int64)&unk_49DB368;
                sub_D68D70(&v40);
                goto LABEL_24;
              }
LABEL_8:
              v37 = v21;
              v39 = (unsigned __int64)&unk_49DB368;
              sub_D68D70(&v40);
              if ( v37 == *(_QWORD *)(a4 + 8) + ((unsigned __int64)*(unsigned int *)(a4 + 24) << 6) )
              {
LABEL_24:
                v39 = 6;
                v40 = 0;
                v41 = 0;
LABEL_25:
                result = sub_D68D70(&v39);
                v16 = *(_QWORD *)(v16 + 8);
                if ( v15 == v16 )
                  return result;
              }
              else
              {
                v39 = 6;
                v40 = 0;
                v41 = *(_BYTE **)(v37 + 56);
                v23 = v41;
                if ( v41 + 4096 != 0 && v41 != 0 && v41 != (_BYTE *)-8192LL )
                {
                  sub_BD6050(&v39, *(_QWORD *)(v37 + 40) & 0xFFFFFFFFFFFFFFF8LL);
                  v23 = v41;
                }
                if ( !v23 || *v23 <= 0x1Cu )
                  goto LABEL_25;
                sub_D68D70(&v39);
                v24 = v16 - 32;
                if ( a6 )
                  v24 = 0;
                v25 = (_BYTE **)(v16 - 96);
                if ( *(_BYTE *)(v16 - 32) == 26 )
                  v25 = (_BYTE **)(v16 - 64);
                v36 = v24;
                v38 = *a1;
                v26 = sub_D69810(*v25, a4, a5, *a1, a7, a8);
                result = sub_1040630(v38, v23, v26, v36, 0);
                if ( result )
                  result = sub_1041C60(*a1, result, a3, 1);
LABEL_20:
                v16 = *(_QWORD *)(v16 + 8);
                if ( v15 == v16 )
                  return result;
              }
            }
          }
        }
      }
    }
    else
    {
      v29 = 1;
      while ( v14 != -4096 )
      {
        v30 = v29 + 1;
        v12 = (result - 1) & (v29 + v12);
        v13 = (__int64 *)(v9 + 16LL * v12);
        v14 = *v13;
        if ( a2 == *v13 )
          goto LABEL_3;
        v29 = v30;
      }
    }
  }
  return result;
}
