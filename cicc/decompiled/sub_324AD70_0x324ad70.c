// Function: sub_324AD70
// Address: 0x324ad70
//
__int64 __fastcall sub_324AD70(__int64 *a1, __int64 a2, __int16 a3, const void *a4, size_t a5)
{
  __int64 result; // rax
  __int16 v9; // r15
  unsigned __int64 v10; // r14
  __int64 v11; // rdi
  unsigned __int64 v12; // rax
  unsigned int v13; // eax
  unsigned __int64 **v14; // r8
  __int64 v15; // rax
  __int64 *v16; // r8
  _QWORD *v17; // r15
  void *v18; // rcx
  const void *v19; // r9
  unsigned int v20; // r12d
  char *v21; // rcx
  void *v22; // rax
  __int64 v23; // rax
  unsigned int v24; // ebx
  __int64 *v27; // [rsp-50h] [rbp-50h]
  __int64 *v28; // [rsp-50h] [rbp-50h]
  __int64 v29; // [rsp-48h] [rbp-48h] BYREF
  _QWORD *v30; // [rsp-40h] [rbp-40h]

  result = a1[10];
  if ( *(_DWORD *)(result + 32) == 3 )
    return result;
  if ( !*(_BYTE *)(a1[26] + 3687) )
  {
    if ( (*(unsigned __int8 (__fastcall **)(__int64 *))(*a1 + 96))(a1) )
    {
      v9 = 7938;
    }
    else
    {
      v9 = 14;
      if ( !*(_BYTE *)(a1[26] + 3770) )
      {
        v9 = 14;
        v10 = sub_3247180(a1[27] + 176, a1[23], a4, a5);
LABEL_6:
        v11 = a1[26];
        if ( *(_BYTE *)(v11 + 3770) )
        {
          v9 = 40;
          v12 = (v10 & 0xFFFFFFFFFFFFFFF8LL) + 8;
          if ( (v10 & 4) != 0 )
            v12 = v10 & 0xFFFFFFFFFFFFFFF8LL;
          v13 = *(_DWORD *)(v12 + 16);
          if ( v13 <= 0xFFFFFF )
          {
            v9 = 39;
            if ( v13 <= 0xFFFF )
              v9 = (v13 > 0xFF) + 37;
          }
        }
        v14 = (unsigned __int64 **)(a2 + 8);
        if ( !a3
          || (*(_BYTE *)(*(_QWORD *)(a1[23] + 200) + 904LL) & 0x40) == 0
          || (v20 = (unsigned __int16)sub_3220AA0(v11),
              result = sub_E06A90(a3),
              v14 = (unsigned __int64 **)(a2 + 8),
              v20 >= (unsigned int)result) )
        {
          WORD2(v29) = a3;
          LODWORD(v29) = 2;
          HIWORD(v29) = v9;
          v30 = (_QWORD *)v10;
          return sub_3248F80(v14, a1 + 11, &v29);
        }
        return result;
      }
    }
    v10 = sub_3247190(a1[27] + 176, a1[23], a4, a5);
    goto LABEL_6;
  }
  v15 = sub_A777F0(0x10u, a1 + 11);
  v16 = a1 + 11;
  v17 = (_QWORD *)v15;
  if ( v15 )
  {
    v18 = 0;
    v19 = a4;
    if ( a5 )
    {
      v21 = (char *)a1[11];
      a1[21] += a5;
      if ( a1[12] >= (unsigned __int64)&v21[a5] && v21 )
      {
        a1[11] = (__int64)&v21[a5];
      }
      else
      {
        v23 = sub_9D1E70((__int64)v16, a5, a5, 0);
        v16 = a1 + 11;
        v19 = a4;
        v21 = (char *)v23;
      }
      v27 = v16;
      v22 = memmove(v21, v19, a5);
      v16 = v27;
      v18 = v22;
    }
    *v17 = v18;
    v17[1] = a5;
  }
  if ( !a3
    || (*(_BYTE *)(*(_QWORD *)(a1[23] + 200) + 904LL) & 0x40) == 0
    || (v28 = v16,
        v24 = (unsigned __int16)sub_3220AA0(a1[26]),
        result = sub_E06A90(a3),
        v16 = v28,
        v24 >= (unsigned int)result) )
  {
    WORD2(v29) = a3;
    v30 = v17;
    LODWORD(v29) = 11;
    HIWORD(v29) = 8;
    return sub_3248F80((unsigned __int64 **)(a2 + 8), v16, &v29);
  }
  return result;
}
