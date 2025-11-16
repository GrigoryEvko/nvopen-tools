// Function: sub_980260
// Address: 0x980260
//
int __fastcall sub_980260(__int64 a1, _BYTE *a2, size_t a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rax
  size_t v8; // rdx
  __int64 v9; // rbx
  size_t v10; // r14
  char v11; // rdx^4
  void *s2; // [rsp+8h] [rbp-58h]
  void *v14; // [rsp+10h] [rbp-50h] BYREF
  size_t n; // [rsp+18h] [rbp-48h]
  __int64 v16; // [rsp+20h] [rbp-40h]
  __int64 v17; // [rsp+28h] [rbp-38h]

  v7 = (__int64)sub_97E150(a2, a3);
  *(_DWORD *)a5 = 0;
  n = v8;
  *(_BYTE *)(a5 + 4) = 1;
  v14 = (void *)v7;
  *(_DWORD *)a4 = 1;
  *(_BYTE *)(a4 + 4) = 0;
  if ( v8 )
  {
    v7 = sub_97F810((__int64 *)(a1 + 176), &v14, (unsigned __int8 (__fastcall *)(__int64, _QWORD, _QWORD))sub_97E7A0);
    v9 = v7;
    if ( v7 != *(_QWORD *)(a1 + 184) )
    {
      LODWORD(v7) = (_DWORD)v14;
      v10 = n;
      s2 = v14;
      while ( 1 )
      {
        if ( *(_QWORD *)(v9 + 8) != v10 )
          return v7;
        if ( v10 )
        {
          LODWORD(v7) = memcmp(*(const void **)v9, s2, v10);
          if ( (_DWORD)v7 )
            return v7;
        }
        if ( *(_BYTE *)(v9 + 36) )
          break;
        v7 = a4;
        if ( !*(_BYTE *)(a4 + 4) )
          goto LABEL_6;
LABEL_8:
        v9 += 64;
        if ( *(_QWORD *)(a1 + 184) == v9 )
          return v7;
      }
      v7 = a5;
LABEL_6:
      if ( *(_DWORD *)(v9 + 32) > *(_DWORD *)v7 )
      {
        v17 = *(_QWORD *)(v9 + 32);
        v11 = BYTE4(v17);
        v16 = v17;
        *(_DWORD *)v7 = v17;
        *(_BYTE *)(v7 + 4) = v11;
      }
      goto LABEL_8;
    }
  }
  return v7;
}
