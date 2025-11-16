// Function: sub_7132C0
// Address: 0x7132c0
//
__int64 __fastcall sub_7132C0(const __m128i *a1, __int64 a2, __int64 a3, int a4, unsigned int *a5, unsigned __int8 *a6)
{
  __int64 result; // rax
  int v10; // eax
  int v11; // r8d
  int v12; // eax
  __int64 v13; // rax
  unsigned int v14; // ecx
  int v15; // r15d
  signed int v16; // [rsp+10h] [rbp-70h]
  int v17; // [rsp+10h] [rbp-70h]
  int v19; // [rsp+18h] [rbp-68h]
  int v20; // [rsp+18h] [rbp-68h]
  int v21; // [rsp+18h] [rbp-68h]
  char v22; // [rsp+23h] [rbp-5Dh] BYREF
  int v23; // [rsp+24h] [rbp-5Ch] BYREF
  int v24; // [rsp+28h] [rbp-58h] BYREF
  int v25; // [rsp+2Ch] [rbp-54h] BYREF
  __m128i i; // [rsp+30h] [rbp-50h] BYREF
  __int16 v27[32]; // [rsp+40h] [rbp-40h] BYREF

  *a5 = 0;
  *a6 = 5;
  sub_7131E0(a2, a1[8].m128i_i64[0], a5);
  result = *a5;
  if ( (_DWORD)result )
  {
    if ( (_DWORD)result != 63 )
      goto LABEL_17;
    result = (__int64)&qword_4F077B4 + 4;
    if ( !HIDWORD(qword_4F077B4) )
      goto LABEL_17;
    if ( *a6 == 8 )
      return result;
    v13 = a1[8].m128i_i64[0];
    for ( i = _mm_loadu_si128(a1 + 11); *(_BYTE *)(v13 + 140) == 12; v13 = *(_QWORD *)(v13 + 160) )
      ;
    v14 = dword_4F06BA0 * *(_DWORD *)(v13 + 128);
    if ( !unk_4F06998 )
    {
      if ( a4 )
      {
        v20 = v14 - 1;
        v15 = sub_620E90((__int64)a1);
        if ( v15 )
        {
          if ( dword_4F0699C )
          {
            sub_6214E0(i.m128i_i16, v20, v15, dword_4F0699C);
          }
          else
          {
            sub_70FEF0(a3, &v22, &v24, &v25);
            sub_621EE0(v27, v25);
            sub_6213D0((__int64)&i, (__int64)v27);
            sub_6214E0(i.m128i_i16, v20, v15, dword_4F0699C);
          }
        }
        else
        {
          sub_6214E0(i.m128i_i16, v20, 0, dword_4F0699C);
        }
        sub_6214E0(i.m128i_i16, 1, v15, dword_4F0699C);
      }
      else
      {
        sub_621410((__int64)&i, v14 - 1, &v23);
        sub_621410((__int64)&i, 1, &v23);
      }
      return sub_70FF50(&i, a3, 0, 0, a5, a6);
    }
    v16 = dword_4F06BA0 * *(_DWORD *)(v13 + 128);
    result = sub_620FA0(a2, &v23);
    if ( v23 )
    {
LABEL_17:
      *a6 = 8;
      return result;
    }
    v11 = (int)result % v16;
    if ( a4 )
      goto LABEL_4;
    sub_621410((__int64)&i, (int)result % v16, &v23);
  }
  else
  {
    if ( *a6 == 8 )
      return result;
    i = _mm_loadu_si128(a1 + 11);
    v10 = sub_620FA0(a2, &v23);
    v11 = v10;
    if ( a4 )
    {
LABEL_4:
      v19 = v11;
      v12 = sub_620E90((__int64)a1);
      if ( v12 )
      {
        if ( dword_4F0699C )
        {
          sub_6214E0(i.m128i_i16, v19, v12, dword_4F0699C);
        }
        else
        {
          v17 = v19;
          v21 = v12;
          sub_70FEF0(a3, &v22, &v24, &v25);
          sub_621EE0(v27, v25);
          sub_6213D0((__int64)&i, (__int64)v27);
          sub_6214E0(i.m128i_i16, v17, v21, dword_4F0699C);
        }
      }
      else
      {
        sub_6214E0(i.m128i_i16, v19, 0, dword_4F0699C);
      }
      return sub_70FF50(&i, a3, 0, 0, a5, a6);
    }
    sub_621410((__int64)&i, v10, &v23);
  }
  return sub_70FF50(&i, a3, 0, 0, a5, a6);
}
