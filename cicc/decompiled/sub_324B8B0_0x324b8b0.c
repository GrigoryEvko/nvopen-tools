// Function: sub_324B8B0
// Address: 0x324b8b0
//
__int64 __fastcall sub_324B8B0(_QWORD *a1, unsigned __int64 **a2, __int16 a3, __int64 a4, __int64 a5)
{
  unsigned __int64 **v5; // r9
  __int64 *v8; // r8
  __int64 v10; // rax
  _QWORD *v11; // rbx
  __int64 result; // rax
  unsigned int v13; // r12d
  __int64 v14; // rax
  unsigned __int64 **v15; // [rsp+0h] [rbp-50h]
  __int64 *v16; // [rsp+8h] [rbp-48h]
  __int64 v17[8]; // [rsp+10h] [rbp-40h] BYREF

  v5 = a2;
  v8 = a1 + 11;
  v10 = a1[11];
  a1[21] += 16LL;
  v11 = (_QWORD *)((v10 + 15) & 0xFFFFFFFFFFFFFFF0LL);
  if ( a1[12] < (unsigned __int64)(v11 + 2) || !v10 )
  {
    v14 = sub_9D1E70((__int64)(a1 + 11), 16, 16, 4);
    v5 = a2;
    v8 = a1 + 11;
    v11 = (_QWORD *)v14;
    goto LABEL_4;
  }
  a1[11] = v11 + 2;
  if ( v11 )
  {
LABEL_4:
    *v11 = a4;
    v11[1] = a5;
  }
  if ( !a3
    || (*(_BYTE *)(*(_QWORD *)(a1[23] + 200LL) + 904LL) & 0x40) == 0
    || (v15 = v5,
        v16 = v8,
        v13 = (unsigned __int16)sub_3220AA0(a1[26]),
        result = sub_E06A90(a3),
        v8 = v16,
        v5 = v15,
        v13 >= (unsigned int)result) )
  {
    LODWORD(v17[0]) = 6;
    WORD2(v17[0]) = a3;
    HIWORD(v17[0]) = 6;
    v17[1] = (__int64)v11;
    return sub_3248F80(v5, v8, v17);
  }
  return result;
}
