// Function: sub_88F760
// Address: 0x88f760
//
__int64 __fastcall sub_88F760(char a1, _QWORD *a2, __int64 a3, _QWORD *a4, const __m128i *a5)
{
  _QWORD *v7; // r12
  const __m128i *v8; // rdi
  __int64 v9; // rax
  _QWORD **v10; // r13
  __int64 *v11; // r15
  const __m128i *v12; // r8
  __int64 *v13; // rax
  const __m128i *v14; // r15
  __int64 result; // rax
  const __m128i *v17; // [rsp+8h] [rbp-48h]
  char v18[49]; // [rsp+1Fh] [rbp-31h] BYREF

  v7 = sub_7271D0();
  *v7 = a2[3];
  v7[2] = sub_87D510(a3, v18);
  *((_BYTE *)v7 + 8) = v18[0];
  if ( a1 == 16 )
    *((_BYTE *)v7 + 24) = 1;
  v7[5] = sub_729420(1, a5);
  v8 = (const __m128i *)a2[23];
  if ( v8 || a2[25] || a2[27] )
  {
    v9 = sub_5CF190(v8);
    v10 = (_QWORD **)(v7 + 4);
    v7[4] = v9;
    v11 = v7 + 4;
    v12 = (const __m128i *)a2[25];
    if ( v9 )
    {
      v17 = (const __m128i *)a2[25];
      v13 = (__int64 *)sub_5CB9F0((_QWORD **)v7 + 4);
      v12 = v17;
      v11 = v13;
    }
    *v11 = sub_5CF190(v12);
    v14 = (const __m128i *)a2[27];
    if ( v7[4] )
      v10 = sub_5CB9F0((_QWORD **)v7 + 4);
    *v10 = (_QWORD *)sub_5CF190(v14);
  }
  result = dword_4F04C3C;
  if ( !dword_4F04C3C )
    return sub_8699D0((__int64)v7, 56, a4);
  return result;
}
