// Function: sub_13523B0
// Address: 0x13523b0
//
void __fastcall sub_13523B0(const char *src, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rax
  _WORD *v8; // rdx
  __int64 v9; // r12
  size_t v10; // rax
  _WORD *v11; // rdi
  size_t v12; // r15
  unsigned __int64 v13; // rax
  __int64 v14; // rdx
  _BYTE *v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax

  if ( byte_4F97AA0 || (_BYTE)a2 )
  {
    v7 = sub_16E8CB0(src, a2, a3);
    v8 = *(_WORD **)(v7 + 24);
    v9 = v7;
    if ( *(_QWORD *)(v7 + 16) - (_QWORD)v8 <= 1u )
    {
      v9 = sub_16E7EE0(v7, "  ", 2);
    }
    else
    {
      *v8 = 8224;
      *(_QWORD *)(v7 + 24) += 2LL;
    }
    v10 = strlen(src);
    v11 = *(_WORD **)(v9 + 24);
    v12 = v10;
    v13 = *(_QWORD *)(v9 + 16) - (_QWORD)v11;
    if ( v12 > v13 )
    {
      v16 = sub_16E7EE0(v9, src, v12);
      v11 = *(_WORD **)(v16 + 24);
      v9 = v16;
      v13 = *(_QWORD *)(v16 + 16) - (_QWORD)v11;
    }
    else if ( v12 )
    {
      memcpy(v11, src, v12);
      v17 = *(_QWORD *)(v9 + 16);
      v11 = (_WORD *)(v12 + *(_QWORD *)(v9 + 24));
      *(_QWORD *)(v9 + 24) = v11;
      v13 = v17 - (_QWORD)v11;
    }
    if ( v13 <= 1 )
    {
      v9 = sub_16E7EE0(v9, ": ", 2);
    }
    else
    {
      *v11 = 8250;
      *(_QWORD *)(v9 + 24) += 2LL;
    }
    sub_155C2B0(a3 & 0xFFFFFFFFFFFFFFF8LL, v9, 0);
    v14 = *(_QWORD *)(v9 + 24);
    if ( (unsigned __int64)(*(_QWORD *)(v9 + 16) - v14) <= 4 )
    {
      v9 = sub_16E7EE0(v9, " <-> ", 5);
    }
    else
    {
      *(_DWORD *)v14 = 1043151904;
      *(_BYTE *)(v14 + 4) = 32;
      *(_QWORD *)(v9 + 24) += 5LL;
    }
    sub_155C2B0(a4 & 0xFFFFFFFFFFFFFFF8LL, v9, 0);
    v15 = *(_BYTE **)(v9 + 24);
    if ( (unsigned __int64)v15 >= *(_QWORD *)(v9 + 16) )
    {
      sub_16E7DE0(v9, 10);
    }
    else
    {
      *(_QWORD *)(v9 + 24) = v15 + 1;
      *v15 = 10;
    }
  }
}
