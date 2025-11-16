// Function: sub_1E0D450
// Address: 0x1e0d450
//
char *__fastcall sub_1E0D450(_QWORD *a1, __int64 a2, __int64 a3, unsigned __int64 a4, _QWORD *a5, int a6)
{
  char *v8; // rax
  char *v9; // rbx
  char *v10; // r15
  __int64 v11; // r15
  __int64 v12; // r14
  int v13; // eax
  _BYTE *v14; // rsi
  char *result; // rax
  __int64 v16; // [rsp+8h] [rbp-68h]
  int v17; // [rsp+1Ch] [rbp-54h] BYREF
  char *v18; // [rsp+20h] [rbp-50h] BYREF
  char *v19; // [rsp+28h] [rbp-48h]
  char *v20; // [rsp+30h] [rbp-40h]

  v16 = sub_1E0C9D0(a1, a2, a3, a4, a5, a6);
  if ( a4 > 0x1FFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
  v18 = 0;
  v19 = 0;
  v20 = 0;
  if ( a4 )
  {
    v8 = (char *)sub_22077B0(4 * a4);
    v18 = v8;
    v9 = v8;
    v10 = &v8[4 * a4];
    v20 = v10;
    if ( v8 != v10 )
      memset(v8, 0, 4 * a4);
    v19 = v10;
    if ( (_DWORD)a4 )
    {
      v11 = 0;
      v12 = 4LL * (unsigned int)(a4 - 1);
      while ( 1 )
      {
        *(_DWORD *)&v9[v11] = sub_1E0C1F0(a1, *(_QWORD *)(a3 + 2 * v11));
        if ( v12 == v11 )
          break;
        v9 = v18;
        v11 += 4;
      }
    }
  }
  v13 = sub_1E0CFC0(a1, (__int64)&v18);
  v17 = v13;
  v14 = *(_BYTE **)(v16 + 104);
  if ( v14 == *(_BYTE **)(v16 + 112) )
  {
    result = sub_1E0CD40(v16 + 96, v14, &v17);
  }
  else
  {
    if ( v14 )
    {
      *(_DWORD *)v14 = v13;
      v14 = *(_BYTE **)(v16 + 104);
    }
    result = (char *)v16;
    *(_QWORD *)(v16 + 104) = v14 + 4;
  }
  if ( v18 )
    return (char *)j_j___libc_free_0(v18, v20 - v18);
  return result;
}
