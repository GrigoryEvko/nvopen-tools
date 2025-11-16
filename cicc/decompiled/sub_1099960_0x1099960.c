// Function: sub_1099960
// Address: 0x1099960
//
bool __fastcall sub_1099960(__int64 a1, char *a2, unsigned __int64 a3)
{
  unsigned __int64 v3; // r14
  __int64 v5; // rax
  __int64 v6; // rbx
  char *v7; // r12
  _QWORD *v8; // r14
  _QWORD *v9; // r13
  int v10; // eax

  v3 = *(_QWORD *)(a1 + 8);
  if ( v3 > a3 || v3 && memcmp(a2, *(const void **)a1, *(_QWORD *)(a1 + 8)) )
  {
LABEL_9:
    LOBYTE(v10) = 0;
  }
  else
  {
    v5 = *(unsigned int *)(a1 + 24);
    v6 = a3 - v3;
    v7 = &a2[v3];
    if ( (_DWORD)v5 )
    {
      v8 = *(_QWORD **)(a1 + 16);
      v9 = &v8[5 * v5];
      while ( 1 )
      {
        LOBYTE(v10) = sub_10997E0(v8, v7, v6);
        if ( (_BYTE)v10 )
          break;
        v8 += 5;
        if ( v9 == v8 )
          goto LABEL_9;
      }
    }
    else
    {
      return v6 == 0;
    }
  }
  return v10;
}
