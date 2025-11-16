// Function: sub_5DB980
// Address: 0x5db980
//
void __fastcall sub_5DB980(FILE *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  bool v7; // bl
  unsigned __int8 v8; // al
  __int64 v9; // rax
  char v10; // al
  __int64 v11; // r13
  __int64 v12; // rdi
  char i; // al
  _QWORD *v14; // rax
  _QWORD *v15; // rdx
  _BOOL4 v16; // ebx
  int v17; // eax
  FILE *v18; // r15
  int v19; // edi
  char *v20; // rbx
  int v21; // edi
  char *v22; // rbx
  __int64 *v23; // rbx
  __int64 *v24; // r14
  __int64 *v25; // r13
  __int64 v26; // rax
  __int64 v27; // rsi
  int v28; // edi
  char *v29; // rbx
  __int64 v30; // r13
  __int64 v31; // rsi
  __int64 v32; // rax

  v6 = (__int64)a1;
  v7 = (a1[-1]._unused2[12] & 0x10) != 0;
  if ( (BYTE5(a1->_lock) & 2) != 0 && (_DWORD)a2 == 1 && sub_5D76E0() )
  {
    v18 = stream;
    sub_5D3B20(qword_4CF7EB0);
    sub_5D6FC0();
    a1 = v18;
    sub_5D3B20(v18);
  }
  if ( !v7 )
  {
    v8 = *(_BYTE *)(v6 + 140);
    if ( v8 > 0xBu )
    {
      if ( v8 != 12 )
        goto LABEL_64;
      v10 = *(_BYTE *)(v6 + 184);
      if ( (unsigned __int8)(v10 - 5) > 2u && v10 != 1 )
      {
        if ( (*(_BYTE *)(v6 - 8) & 0x20) != 0 )
        {
          *(_BYTE *)(v6 + 142) |= 0x40u;
        }
        else if ( (_DWORD)a2 == 2 )
        {
          v11 = v6;
          while ( 1 )
          {
            v12 = v11;
            if ( !(unsigned int)sub_8D3410(v11) )
              break;
LABEL_22:
            v11 = sub_8D40F0(v12);
            for ( i = *(_BYTE *)(v11 + 140); i == 12; i = *(_BYTE *)(v11 + 140) )
              v11 = *(_QWORD *)(v11 + 160);
            if ( (unsigned __int8)(i - 9) <= 2u && (*(_BYTE *)(v11 + 142) & 0x20) == 0 )
            {
              v14 = (_QWORD *)qword_4CF7CB8;
              if ( qword_4CF7CB8 )
                qword_4CF7CB8 = *(_QWORD *)qword_4CF7CB8;
              else
                v14 = (_QWORD *)sub_822B10(24);
              v15 = (_QWORD *)qword_4CF7CC0;
              *v14 = 0;
              v14[1] = v6;
              v14[2] = v11;
              if ( v15 )
                *v15 = v14;
              else
                qword_4CF7CC8 = (__int64)v14;
              qword_4CF7CC0 = (__int64)v14;
              *(_BYTE *)(v6 + 142) |= 8u;
              *(_BYTE *)(v11 + 142) |= 8u;
              goto LABEL_31;
            }
          }
          while ( (unsigned int)sub_8D2E30(v12) )
          {
            v11 = sub_8D46C0(v11);
            v12 = v11;
            if ( (unsigned int)sub_8D3410(v11) )
              goto LABEL_22;
          }
          sub_5D6C90(v6, a2);
LABEL_31:
          if ( (*(_BYTE *)(v6 + 141) & 1) != 0 && !dword_4CF7C78 )
          {
            if ( qword_4CF7EB8 == stream )
            {
              sub_5D3EB0("#include \"crt/device_runtime.h\"");
              dword_4CF7C78 = 1;
            }
            else
            {
              sub_5D3EB0("#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)");
              sub_5D3EB0("#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__");
              sub_5D3EB0("#endif");
              sub_5D3EB0("#include \"crt/host_runtime.h\"");
            }
          }
        }
      }
    }
    else
    {
      if ( v8 <= 9u )
      {
        if ( v8 == 2 )
        {
          if ( (**(_BYTE **)(v6 + 176) & 1) != 0 )
          {
            v9 = *(_QWORD *)(v6 + 168);
            if ( (*(_BYTE *)(v6 + 161) & 0x10) != 0 )
              v9 = *(_QWORD *)(v9 + 96);
            if ( v9 )
            {
              if ( (_DWORD)a2 == 1 )
                sub_5D72F0(v6, 1, a3, a4, a5, a6);
            }
          }
          return;
        }
LABEL_64:
        sub_721090(a1);
      }
      v16 = 0;
      v17 = sub_8D23B0(v6);
      if ( (*(_BYTE *)(v6 + 178) & 0x40) != 0 )
        v16 = v17 == 0;
      if ( unk_4F068C4
        && unk_4F06A80 | unk_4F06A7C | unk_4F06A78
        && (*(_BYTE *)(*(_QWORD *)(v6 + 168) + 110LL) & 0x40) != 0 )
      {
        if ( (_DWORD)a2 == 2 && v16 )
        {
          v19 = 116;
          v20 = "ypedef typeof(((__builtin_va_list*)0)[0][0]) ";
          do
          {
            ++v20;
            putc(v19, stream);
            v19 = *(v20 - 1);
          }
          while ( *(v20 - 1) );
          dword_4CF7F40 += 46;
          v21 = 95;
          v22 = "_va_list_tag_type;";
          do
          {
            ++v22;
            putc(v21, stream);
            v21 = *(v22 - 1);
          }
          while ( *(v22 - 1) );
          dword_4CF7F40 += 19;
        }
      }
      else if ( (_DWORD)a2 == 1 )
      {
        if ( (*(_BYTE *)(v6 + 88) & 8) == 0 )
        {
          if ( !dword_4CF7EA0 )
            return;
          sub_5D3EB0("#if 0");
          if ( dword_4CF7F40 )
            sub_5D37C0("#if 0", a2);
          dword_4CF7F3C = 0;
          dword_4CF7F44 = 0;
          qword_4CF7F48 = 0;
          dword_4F07508[0] = 0;
          LOWORD(dword_4F07508[1]) = 0;
        }
        if ( !v16 && *(char *)(v6 + 88) < 0 )
        {
          v30 = 0;
          while ( 1 )
          {
            v31 = unk_4F04C50;
            if ( unk_4F04C50 )
              v31 = qword_4CF7E98;
            v32 = sub_732D20(v6, v31, 0, v30);
            v30 = v32;
            if ( !v32 )
              break;
            sub_5D52E0(v32, v31);
          }
        }
        sub_5D45D0((unsigned int *)(v6 + 64));
        sub_5DB710(v6);
        if ( (unsigned int)sub_8D23B0(v6) && (*(_BYTE *)(v6 + 180) & 0x20) != 0 )
        {
          v28 = 32;
          v29 = "{ char dummy; }";
          do
          {
            ++v29;
            putc(v28, stream);
            v28 = *(v29 - 1);
            ++dword_4CF7F40;
          }
          while ( (_BYTE)v28 );
        }
        putc(59, stream);
        v27 = (unsigned int)dword_4CF7EA0;
        ++dword_4CF7F40;
        if ( dword_4CF7EA0 && (*(_BYTE *)(v6 + 88) & 8) == 0 )
        {
          sub_5D3EB0("#endif");
          if ( dword_4CF7F40 )
            sub_5D37C0("#endif", v27);
          dword_4CF7F3C = 0;
          dword_4CF7F44 = 0;
          qword_4CF7F48 = 0;
          dword_4F07508[0] = 0;
          LOWORD(dword_4F07508[1]) = 0;
        }
      }
      else if ( v16 )
      {
        sub_5DAD30((const char *)v6, 1);
        if ( (*(_BYTE *)(v6 + 142) & 8) != 0 )
        {
          v23 = (__int64 *)qword_4CF7CC8;
          if ( qword_4CF7CC8 )
          {
            v24 = 0;
            do
            {
              v25 = v23;
              v23 = (__int64 *)*v23;
              if ( v6 == v25[2] )
              {
                sub_5D6C90(v25[1], 1);
                *(_BYTE *)(v25[1] + 142) &= ~8u;
                if ( v24 )
                  *v24 = *v25;
                else
                  qword_4CF7CC8 = *v25;
                if ( (__int64 *)qword_4CF7CC0 == v25 )
                  qword_4CF7CC0 = (__int64)v24;
                v26 = qword_4CF7CB8;
                qword_4CF7CB8 = (__int64)v25;
                *v25 = v26;
              }
              else
              {
                v24 = v25;
              }
            }
            while ( v23 );
          }
          *(_BYTE *)(v6 + 142) &= ~8u;
        }
      }
    }
  }
}
