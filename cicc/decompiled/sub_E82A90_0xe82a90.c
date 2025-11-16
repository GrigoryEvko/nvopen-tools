// Function: sub_E82A90
// Address: 0xe82a90
//
void __fastcall sub_E82A90(__int64 a1, __int64 a2, _QWORD *a3, int a4, char a5, __int64 a6)
{
  unsigned int v9; // r15d
  __int64 v10; // r8
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rdx
  _QWORD *v14; // rdx
  __int64 v15; // rdx
  size_t v16; // rdx
  char *v17; // rsi
  int v18; // [rsp+Ch] [rbp-34h]

  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 8) = a3;
  *(_BYTE *)(a1 + 16) = a5;
  *(_BYTE *)(a1 + 17) = a6;
  if ( (_BYTE)a6 )
  {
    if ( (unsigned int)a4 > 3 )
    {
      v10 = 17;
      v9 = 17;
    }
    else
    {
      v9 = dword_3F807D0[a4];
      v10 = v9;
    }
    v11 = *(unsigned int *)(a2 + 72);
    if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 76) )
    {
      v18 = v10;
      sub_C8D5F0(a2 + 64, (const void *)(a2 + 80), v11 + 1, 4u, v10, a6);
      v11 = *(unsigned int *)(a2 + 72);
      LODWORD(v10) = v18;
    }
    *(_DWORD *)(*(_QWORD *)(a2 + 64) + 4 * v11) = v10;
    ++*(_DWORD *)(a2 + 72);
    (*(void (__fastcall **)(_QWORD *, _QWORD, _QWORD, _QWORD))(*a3 + 24LL))(a3, v9, 0, 0);
  }
  if ( a5 )
  {
    if ( a4 == 2 )
    {
      v14 = (_QWORD *)a3[4];
      if ( a3[3] - (_QWORD)v14 > 7u )
      {
        *v14 = 0x3A7465677261743CLL;
        a3[4] += 8LL;
        return;
      }
      v16 = 8;
      v17 = "<target:";
      goto LABEL_24;
    }
    if ( a4 <= 2 )
    {
      if ( a4 )
      {
        if ( a4 != 1 )
          return;
        v12 = a3[4];
        if ( (unsigned __int64)(a3[3] - v12) > 4 )
        {
          *(_DWORD *)v12 = 1734701628;
          *(_BYTE *)(v12 + 4) = 58;
          a3[4] += 5LL;
          return;
        }
        v16 = 5;
        v17 = "<reg:";
      }
      else
      {
        v15 = a3[4];
        if ( (unsigned __int64)(a3[3] - v15) > 4 )
        {
          *(_DWORD *)v15 = 1835886908;
          *(_BYTE *)(v15 + 4) = 58;
          a3[4] += 5LL;
          return;
        }
        v16 = 5;
        v17 = "<imm:";
      }
LABEL_24:
      sub_CB6200((__int64)a3, (unsigned __int8 *)v17, v16);
      return;
    }
    if ( a4 == 3 )
    {
      v13 = a3[4];
      if ( (unsigned __int64)(a3[3] - v13) > 4 )
      {
        *(_DWORD *)v13 = 1835363644;
        *(_BYTE *)(v13 + 4) = 58;
        a3[4] += 5LL;
        return;
      }
      v16 = 5;
      v17 = "<mem:";
      goto LABEL_24;
    }
  }
}
