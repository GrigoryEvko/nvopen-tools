// Function: sub_392BD70
// Address: 0x392bd70
//
__int64 __fastcall sub_392BD70(__int64 a1, _QWORD *a2)
{
  int v2; // eax
  __m128i *v3; // rax
  unsigned __int64 v4; // rdx
  __int64 v6; // rdx
  __int64 v7; // rax
  unsigned __int64 v8; // [rsp+8h] [rbp-48h] BYREF
  unsigned __int64 v9[2]; // [rsp+10h] [rbp-40h] BYREF
  _QWORD v10[6]; // [rsp+20h] [rbp-30h] BYREF

  v2 = sub_392A7D0(a2);
  if ( v2 == 34 )
  {
LABEL_9:
    v6 = a2[13];
    v7 = a2[18];
    *(_DWORD *)a1 = 3;
    *(_DWORD *)(a1 + 32) = 64;
    *(_QWORD *)(a1 + 8) = v6;
    *(_QWORD *)(a1 + 16) = v7 - v6;
    *(_QWORD *)(a1 + 24) = 0;
    return a1;
  }
  while ( v2 != 92 )
  {
    if ( v2 == -1 )
      goto LABEL_7;
LABEL_4:
    v2 = sub_392A7D0(a2);
    if ( v2 == 34 )
      goto LABEL_9;
  }
  if ( (unsigned int)sub_392A7D0(a2) != -1 )
    goto LABEL_4;
LABEL_7:
  v8 = 28;
  v9[0] = (unsigned __int64)v10;
  v3 = (__m128i *)sub_22409D0((__int64)v9, &v8, 0);
  v9[0] = (unsigned __int64)v3;
  v10[0] = v8;
  *v3 = _mm_load_si128((const __m128i *)&xmmword_3F90170);
  v4 = v9[0];
  qmemcpy(&v3[1], "ing constant", 12);
  v9[1] = v8;
  *(_BYTE *)(v4 + v8) = 0;
  sub_392A760(a1, a2, a2[13], v9);
  if ( (_QWORD *)v9[0] != v10 )
  {
    j_j___libc_free_0(v9[0]);
    return a1;
  }
  return a1;
}
