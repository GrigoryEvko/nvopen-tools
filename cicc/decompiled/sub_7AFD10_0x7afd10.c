// Function: sub_7AFD10
// Address: 0x7afd10
//
__int64 __fastcall sub_7AFD10(char *filename, __int64 *a2, __int64 a3, char a4)
{
  unsigned int v4; // r14d
  __int64 *v6; // rax
  __int64 *v7; // r13
  __int64 v8; // r15
  unsigned int v9; // r9d
  __int64 v11; // rdi
  __int64 *v12; // rax
  __int64 v13; // rax
  char *v14; // rax
  __int64 *v15; // [rsp+10h] [rbp-70h]
  char *v16; // [rsp+20h] [rbp-60h] BYREF
  char v17; // [rsp+28h] [rbp-58h]
  __m128i v18[4]; // [rsp+38h] [rbp-48h] BYREF

  v4 = a3;
  v16 = filename;
  v17 = (16 * (a4 & 1)) | v17 & 0xEF;
  v6 = (__int64 *)sub_881B20(qword_4F08510, &v16, a3);
  v7 = v6;
  if ( !v6 || (v8 = *v6, v9 = 1, !*v6) )
  {
    sub_7217C0(filename, (__dev_t *)v18);
    v11 = qword_4F08508;
    if ( !qword_4F08508 )
    {
      qword_4F08508 = sub_881A70(0, 1024, 10, 11);
      v11 = qword_4F08508;
    }
    v12 = (__int64 *)sub_881B20(v11, &v16, v4);
    if ( v12 )
    {
      v8 = *v12;
      if ( v4 )
      {
        if ( v8 )
        {
          *v7 = v8;
          v9 = 1;
          goto LABEL_3;
        }
      }
      else if ( v8 )
      {
        v9 = 1;
        goto LABEL_3;
      }
    }
    v9 = 0;
    v8 = 0;
    if ( v4 )
    {
      v15 = v12;
      v13 = sub_823970(40);
      *(_BYTE *)(v13 + 8) &= 0xE0u;
      v8 = v13;
      *(_QWORD *)v13 = 0;
      *(_QWORD *)(v13 + 16) = 0;
      sub_7217B0((_QWORD *)(v13 + 24));
      v14 = sub_724840(0, filename);
      v9 = 0;
      *(_QWORD *)v8 = v14;
      *(__m128i *)(v8 + 24) = _mm_loadu_si128(v18);
      *v15 = v8;
      *v7 = v8;
    }
  }
LABEL_3:
  *a2 = v8;
  return v9;
}
