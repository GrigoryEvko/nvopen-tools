// Function: sub_9B4A70
// Address: 0x9b4a70
//
__int64 __fastcall sub_9B4A70(__int64 **a1, char a2)
{
  __int64 v2; // rax
  char v3; // dl
  __int64 v4; // rax
  __int64 v5; // r13
  __int64 result; // rax
  __int64 v7; // rdx
  _QWORD *v8; // rdx
  _BYTE *v9; // rdi
  __int64 v10; // rbx
  char *v11; // r14
  __int64 v12; // rax
  __int64 v13; // rdi
  unsigned int v14; // eax

  v2 = **a1;
  v3 = *(_BYTE *)(v2 + 7) & 0x40;
  if ( a2 )
  {
    if ( v3 )
      v4 = *(_QWORD *)(v2 - 8);
    else
      v4 = v2 - 32LL * (*(_DWORD *)(v2 + 4) & 0x7FFFFFF);
    v5 = *(_QWORD *)(v4 + 32);
  }
  else
  {
    if ( v3 )
      v12 = *(_QWORD *)(v2 - 8);
    else
      v12 = v2 - 32LL * (*(_DWORD *)(v2 + 4) & 0x7FFFFFF);
    v5 = *(_QWORD *)(v12 + 64);
  }
  result = sub_9A6530(v5, (__int64)a1[1], (const __m128i *)a1[2], *(_DWORD *)a1[3]);
  if ( !(_BYTE)result )
  {
    v7 = **a1;
    v8 = (*(_BYTE *)(v7 + 7) & 0x40) != 0
       ? *(_QWORD **)(v7 - 8)
       : (_QWORD *)(v7 - 32LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF));
    v9 = (_BYTE *)*v8;
    if ( *(_BYTE *)*v8 == 82 )
    {
      v10 = *((_QWORD *)v9 - 8);
      v11 = (char *)*((_QWORD *)v9 - 4);
      if ( v5 == v10 && v11 )
      {
        v13 = (unsigned int)sub_B53900(v9);
        goto LABEL_20;
      }
      if ( (char *)v5 == v11 && v10 )
      {
        v11 = (char *)*((_QWORD *)v9 - 8);
        v13 = (unsigned int)sub_B53960(v9);
LABEL_20:
        if ( a2 )
        {
          return sub_9867F0(v13, v11);
        }
        else
        {
          v14 = sub_B52870(v13);
          return sub_9867F0(v14, v11);
        }
      }
    }
  }
  return result;
}
