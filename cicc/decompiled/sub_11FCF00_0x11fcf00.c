// Function: sub_11FCF00
// Address: 0x11fcf00
//
void __fastcall sub_11FCF00(_QWORD *a1)
{
  __int64 v1; // rdx
  _BYTE *v2; // r13
  _BYTE *v3; // rdx
  _BYTE *v4; // rbx
  _BYTE *v5; // r12
  __int64 v6; // r15
  int v7; // eax
  int v8; // eax
  int v9; // [rsp-54h] [rbp-54h]
  _BYTE *v10; // [rsp-50h] [rbp-50h]
  unsigned __int64 v11; // [rsp-48h] [rbp-48h]
  unsigned __int64 v12; // [rsp-40h] [rbp-40h]

  v1 = a1[1];
  if ( v1 )
  {
    v2 = (_BYTE *)*a1;
    v3 = (_BYTE *)(*a1 + v1);
    if ( (_BYTE *)*a1 != v3 )
    {
      v4 = (_BYTE *)*a1;
      v5 = (_BYTE *)*a1;
      v12 = (unsigned __int64)(v3 - 1);
      v11 = (unsigned __int64)(v3 - 2);
      while ( 1 )
      {
        while ( 1 )
        {
          ++v5;
          if ( *v4 == 92 )
            break;
          *(v5 - 1) = *v4++;
LABEL_5:
          if ( v4 == v3 )
            goto LABEL_13;
        }
        if ( (unsigned __int64)v4 < v12 && v4[1] == 92 )
        {
          *(v5 - 1) = 92;
          v4 += 2;
          goto LABEL_5;
        }
        if ( (unsigned __int64)v4 >= v11
          || (v6 = (unsigned __int8)v4[1], v10 = v3, v7 = isxdigit((unsigned __int8)v4[1]), v3 = v10, !v7)
          || (v9 = (unsigned __int8)v4[2], v8 = isxdigit(v9), v3 = v10, !v8) )
        {
          *(v5 - 1) = 92;
          ++v4;
          goto LABEL_5;
        }
        v4 += 3;
        *(v5 - 1) = LOBYTE(word_3F64060[v9]) + 16 * word_3F64060[v6];
        if ( v4 == v10 )
          goto LABEL_13;
      }
    }
    v5 = (_BYTE *)*a1;
LABEL_13:
    sub_22410F0(a1, v5 - v2, 0);
  }
}
