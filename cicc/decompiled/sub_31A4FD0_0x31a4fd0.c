// Function: sub_31A4FD0
// Address: 0x31a4fd0
//
__int64 __fastcall sub_31A4FD0(__int64 a1, __int64 a2, unsigned __int8 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v7; // eax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 result; // rax
  bool v13; // zf

  v7 = dword_4F87508[0];
  *(_QWORD *)a1 = "vectorize.width";
  *(_DWORD *)(a1 + 8) = v7;
  *(_QWORD *)(a1 + 16) = "interleave.count";
  *(_QWORD *)(a1 + 32) = "vectorize.enable";
  *(_QWORD *)(a1 + 40) = 0x2FFFFFFFFLL;
  *(_QWORD *)(a1 + 48) = "isvectorized";
  *(_QWORD *)(a1 + 56) = 0x300000000LL;
  *(_QWORD *)(a1 + 64) = "vectorize.predicate.enable";
  *(_QWORD *)(a1 + 72) = 0x4FFFFFFFFLL;
  *(_QWORD *)(a1 + 80) = "vectorize.scalable.enable";
  *(_DWORD *)(a1 + 24) = a3;
  *(_QWORD *)(a1 + 88) = 0x5FFFFFFFFLL;
  *(_QWORD *)(a1 + 104) = a2;
  *(_QWORD *)(a1 + 112) = a4;
  *(_DWORD *)(a1 + 12) = 0;
  *(_DWORD *)(a1 + 28) = 1;
  *(_BYTE *)(a1 + 96) = 0;
  sub_31A4D80(a1, a2, a3, a4, a5, a6);
  if ( sub_D34290() )
    *(_DWORD *)(a1 + 24) = dword_4F87428[0];
  if ( *(_DWORD *)(a1 + 88) != -1 )
  {
LABEL_4:
    result = (unsigned int)dword_5035008;
    if ( dword_5035008 == -1 )
    {
LABEL_6:
      if ( *(_DWORD *)(a1 + 56) == 1 )
        return result;
      if ( *(_DWORD *)(a1 + 88) != 1 && *(_DWORD *)(a1 + 8) == 1 )
      {
        result = *(unsigned int *)(a1 + 24);
        if ( (_DWORD)result )
        {
          LODWORD(v8) = result == 1;
        }
        else
        {
          result = (unsigned int)sub_F6E5D0(*(_QWORD *)(a1 + 104), a2, v8, v9, v10, v11) >> 1;
          LODWORD(v8) = result & 1;
        }
      }
      else
      {
        LODWORD(v8) = 0;
      }
      goto LABEL_10;
    }
LABEL_5:
    *(_DWORD *)(a1 + 88) = result;
    goto LABEL_6;
  }
  if ( a5 )
    *(_DWORD *)(a1 + 88) = (unsigned __int8)sub_DFE640(a5);
  v8 = *(unsigned int *)(a1 + 8);
  if ( (_DWORD)v8 )
  {
    *(_DWORD *)(a1 + 88) = 0;
    goto LABEL_4;
  }
  result = (unsigned int)dword_5035008;
  if ( dword_5035008 != -1 )
    goto LABEL_5;
  if ( *(_DWORD *)(a1 + 88) != -1 )
    goto LABEL_6;
  v13 = *(_DWORD *)(a1 + 56) == 1;
  *(_DWORD *)(a1 + 88) = 0;
  if ( !v13 )
LABEL_10:
    *(_DWORD *)(a1 + 56) = v8;
  return result;
}
