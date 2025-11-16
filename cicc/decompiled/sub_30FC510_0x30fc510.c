// Function: sub_30FC510
// Address: 0x30fc510
//
__int64 __fastcall sub_30FC510(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // r15
  __int64 v5; // r13
  __int64 v6; // r12
  __int64 v7; // rbx
  __int64 v8; // rax
  _BYTE *v9; // rsi
  unsigned __int64 v10; // rdx
  char *v11; // rcx
  _BYTE *v12; // rax
  size_t v13; // r15
  __int64 v14; // rax
  _BYTE *v15; // rsi
  __int64 v16; // rdx
  int v17; // eax
  int v18; // eax
  unsigned __int64 v19; // [rsp+8h] [rbp-38h]

  result = 0x7FFFFFFFFFFFFFD0LL;
  v4 = 80 * a3;
  v5 = a2 + 80 * a3;
  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  if ( (unsigned __int64)(80 * a3) > 0x7FFFFFFFFFFFFFD0LL )
    sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
  if ( v4 )
  {
    v6 = a2;
    result = sub_22077B0(80 * a3);
    *a1 = result;
    v7 = result;
    for ( a1[2] = result + v4; v5 != v6; v7 += 80 )
    {
      if ( v7 )
      {
        v15 = *(_BYTE **)v6;
        v16 = *(_QWORD *)(v6 + 8);
        *(_QWORD *)v7 = v7 + 16;
        sub_30FA730((__int64 *)v7, v15, (__int64)&v15[v16]);
        v17 = *(_DWORD *)(v6 + 32);
        v9 = *(_BYTE **)(v6 + 40);
        *(_QWORD *)(v7 + 40) = 0;
        *(_QWORD *)(v7 + 48) = 0;
        *(_DWORD *)(v7 + 32) = v17;
        v18 = *(_DWORD *)(v6 + 36);
        *(_QWORD *)(v7 + 56) = 0;
        *(_DWORD *)(v7 + 36) = v18;
        v12 = *(_BYTE **)(v6 + 48);
        v10 = v12 - v9;
        if ( v12 == v9 )
        {
          v13 = 0;
          v11 = 0;
        }
        else
        {
          if ( v10 > 0x7FFFFFFFFFFFFFF8LL )
            sub_4261EA(v7, v9, v10);
          v19 = *(_QWORD *)(v6 + 48) - (_QWORD)v9;
          v8 = sub_22077B0(v10);
          v9 = *(_BYTE **)(v6 + 40);
          v10 = v19;
          v11 = (char *)v8;
          v12 = *(_BYTE **)(v6 + 48);
          v13 = v12 - v9;
        }
        *(_QWORD *)(v7 + 40) = v11;
        *(_QWORD *)(v7 + 48) = v11;
        *(_QWORD *)(v7 + 56) = &v11[v10];
        if ( v9 != v12 )
          v11 = (char *)memmove(v11, v9, v13);
        v14 = *(_QWORD *)(v6 + 64);
        *(_QWORD *)(v7 + 48) = &v11[v13];
        *(_QWORD *)(v7 + 64) = v14;
        result = *(_QWORD *)(v6 + 72);
        *(_QWORD *)(v7 + 72) = result;
      }
      v6 += 80;
    }
  }
  else
  {
    v7 = 0;
  }
  a1[1] = v7;
  return result;
}
