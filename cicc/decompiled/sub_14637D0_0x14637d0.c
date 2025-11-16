// Function: sub_14637D0
// Address: 0x14637d0
//
unsigned __int64 __fastcall sub_14637D0(__int64 *a1, __int64 a2, __int64 a3, __m128i a4, __m128i a5)
{
  unsigned __int64 result; // rax
  __int64 v7; // r14
  __int64 **v8; // r8
  __int64 v9; // r12
  __int64 v10; // r15
  char *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r9
  char *v15; // r13
  unsigned int v16; // r9d
  unsigned int v17; // r11d
  unsigned int v18; // edi
  __int64 v19; // rcx
  __int64 v20; // rax
  __int64 v21; // r8
  unsigned int v22; // eax
  __int16 v23; // r10
  __int64 v24; // rsi
  __int64 *v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // rdx
  __int64 **v28; // [rsp+0h] [rbp-C0h]
  __int64 v29; // [rsp+8h] [rbp-B8h] BYREF
  __int64 v30; // [rsp+10h] [rbp-B0h]
  int v31[2]; // [rsp+18h] [rbp-A8h]
  __int64 v32; // [rsp+20h] [rbp-A0h]
  __int64 v33; // [rsp+28h] [rbp-98h]
  __int64 v34; // [rsp+30h] [rbp-90h] BYREF
  int v35; // [rsp+38h] [rbp-88h] BYREF
  __int64 v36; // [rsp+40h] [rbp-80h]
  int *v37; // [rsp+48h] [rbp-78h]
  int *v38; // [rsp+50h] [rbp-70h]
  __int64 v39; // [rsp+58h] [rbp-68h]
  __int64 v40; // [rsp+60h] [rbp-60h] BYREF
  int v41; // [rsp+68h] [rbp-58h] BYREF
  __int64 v42; // [rsp+70h] [rbp-50h]
  int *v43; // [rsp+78h] [rbp-48h]
  int *v44; // [rsp+80h] [rbp-40h]
  __int64 v45; // [rsp+88h] [rbp-38h]

  result = *((unsigned int *)a1 + 2);
  v29 = a2;
  if ( result > 1 )
  {
    v35 = 0;
    v37 = &v35;
    v7 = *a1;
    v38 = &v35;
    v36 = 0;
    v39 = 0;
    v41 = 0;
    v42 = 0;
    v43 = &v41;
    v44 = &v41;
    v45 = 0;
    if ( result == 2 )
    {
      if ( (int)sub_1462150(&v34, &v40, v29, *(__int64 **)(v7 + 8), *(_QWORD *)v7, a3, 0) < 0 )
      {
        v27 = *(_QWORD *)(v7 + 8);
        *(_QWORD *)(v7 + 8) = *(_QWORD *)v7;
        *(_QWORD *)v7 = v27;
      }
    }
    else
    {
      v30 = (__int64)&v34;
      *(_QWORD *)v31 = &v40;
      v8 = (__int64 **)(v7 + 8 * result);
      v9 = (__int64)(8 * result) >> 3;
      v32 = (__int64)&v29;
      v33 = a3;
      do
      {
        v10 = 8 * v9;
        v28 = v8;
        v11 = (char *)sub_2207800(8 * v9, &unk_435FF63);
        v8 = v28;
        v15 = v11;
        if ( v11 )
        {
          sub_14636A0(
            (__int64 *)v7,
            v28,
            v11,
            v9,
            (__int64)v28,
            a4,
            a5,
            v14,
            (_QWORD *)v30,
            *(_QWORD **)v31,
            (__int64 *)v32,
            v33);
          goto LABEL_6;
        }
        v9 >>= 1;
      }
      while ( v9 );
      v10 = 0;
      sub_1462E70((__int64 *)v7, v28, v12, v13, (__int64)v28, v14, (_QWORD *)v30, *(_QWORD **)v31, (__int64 *)v32, v33);
LABEL_6:
      j_j___libc_free_0(v15, v10);
      v16 = *((_DWORD *)a1 + 2);
      v17 = v16 - 2;
      if ( v16 != 2 )
      {
        v18 = 0;
        do
        {
          while ( 1 )
          {
            v19 = *a1;
            v20 = v18++;
            v21 = *(_QWORD *)(*a1 + 8 * v20);
            v22 = v18;
            v23 = *(_WORD *)(v21 + 24);
            if ( v18 != v16 )
              break;
            v18 = v16;
            if ( v17 == v16 )
              goto LABEL_17;
          }
          while ( 1 )
          {
            v24 = v19 + 8LL * v22;
            if ( v23 != *(_WORD *)(*(_QWORD *)v24 + 24LL) )
              break;
            if ( *(_QWORD *)v24 == v21 )
            {
              v25 = (__int64 *)(v19 + 8LL * v18);
              v26 = *v25;
              *v25 = v21;
              *(_QWORD *)v24 = v26;
              if ( v17 == v18 )
                goto LABEL_17;
              ++v22;
              ++v18;
              if ( v22 == v16 )
                break;
            }
            else if ( ++v22 == v16 )
            {
              break;
            }
            v19 = *a1;
          }
        }
        while ( v17 != v18 );
      }
    }
LABEL_17:
    sub_1453E60(v42);
    return sub_1454030(v36);
  }
  return result;
}
