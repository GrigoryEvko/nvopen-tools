// Function: sub_39EEA00
// Address: 0x39eea00
//
void (*__fastcall sub_39EEA00(__int64 a1, _BYTE *a2, unsigned int *a3))()
{
  __int64 (*v6)(); // rax
  __int64 v7; // rdi
  __int64 v8; // rdx
  __int64 v9; // rdi
  _WORD *v10; // rdx
  unsigned __int64 v11; // r15
  __int64 v13; // rdi
  _BYTE *v14; // rax
  __int64 v15; // rbx
  char *v16; // rsi
  size_t v17; // rdx
  void *v18; // rdi

  if ( *a3 != 4
    || (v6 = *(__int64 (**)())(*((_QWORD *)a3 - 1) + 48LL), v6 == sub_2162C30)
    || !((unsigned __int8 (__fastcall *)(unsigned int *))v6)(a3 - 2) )
  {
    v7 = *(_QWORD *)(a1 + 272);
    v8 = *(_QWORD *)(v7 + 24);
    if ( (unsigned __int64)(*(_QWORD *)(v7 + 16) - v8) <= 4 )
    {
      sub_16E7EE0(v7, ".set ", 5u);
    }
    else
    {
      *(_DWORD *)v8 = 1952805678;
      *(_BYTE *)(v8 + 4) = 32;
      *(_QWORD *)(v7 + 24) += 5LL;
    }
    sub_38E2490(a2, *(_QWORD *)(a1 + 272), *(_BYTE **)(a1 + 280));
    v9 = *(_QWORD *)(a1 + 272);
    v10 = *(_WORD **)(v9 + 24);
    if ( *(_QWORD *)(v9 + 16) - (_QWORD)v10 <= 1u )
    {
      sub_16E7EE0(v9, ", ", 2u);
    }
    else
    {
      *v10 = 8236;
      *(_QWORD *)(v9 + 24) += 2LL;
    }
    sub_38CDBE0((__int64)a3, *(_QWORD *)(a1 + 272), *(_QWORD *)(a1 + 280));
    v11 = *(unsigned int *)(a1 + 312);
    if ( *(_DWORD *)(a1 + 312) )
    {
      v15 = *(_QWORD *)(a1 + 272);
      v16 = *(char **)(a1 + 304);
      v17 = *(unsigned int *)(a1 + 312);
      v18 = *(void **)(v15 + 24);
      if ( v11 > *(_QWORD *)(v15 + 16) - (_QWORD)v18 )
      {
        sub_16E7EE0(*(_QWORD *)(a1 + 272), v16, v17);
      }
      else
      {
        memcpy(v18, v16, v17);
        *(_QWORD *)(v15 + 24) += v11;
      }
    }
    *(_DWORD *)(a1 + 312) = 0;
    if ( (*(_BYTE *)(a1 + 680) & 1) != 0 )
    {
      sub_39E0440(a1);
    }
    else
    {
      v13 = *(_QWORD *)(a1 + 272);
      v14 = *(_BYTE **)(v13 + 24);
      if ( (unsigned __int64)v14 >= *(_QWORD *)(v13 + 16) )
      {
        sub_16E7DE0(v13, 10);
      }
      else
      {
        *(_QWORD *)(v13 + 24) = v14 + 1;
        *v14 = 10;
      }
    }
  }
  return sub_38DDC10(a1, (__int64)a2, a3);
}
