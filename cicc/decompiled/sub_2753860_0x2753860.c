// Function: sub_2753860
// Address: 0x2753860
//
__int64 __fastcall sub_2753860(__int64 a1, __int64 a2, unsigned __int8 *a3)
{
  int v4; // eax
  __int64 v5; // rax
  __int64 v6; // rax
  unsigned __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rbx
  __int64 v11; // r14
  unsigned int v12; // r15d
  unsigned __int64 v13; // rdx
  __int64 v14; // rax

  v4 = *a3;
  if ( (_BYTE)v4 != 85 )
  {
    v8 = (unsigned int)(v4 - 34);
    if ( (unsigned __int8)v8 > 0x33u )
      goto LABEL_9;
    v9 = 0x8000000000041LL;
    if ( !_bittest64(&v9, v8) )
      goto LABEL_9;
    goto LABEL_5;
  }
  v5 = *((_QWORD *)a3 - 4);
  if ( !v5 )
    goto LABEL_5;
  if ( *(_BYTE *)v5 )
    goto LABEL_5;
  if ( *(_QWORD *)(v5 + 24) != *((_QWORD *)a3 + 10) )
    goto LABEL_5;
  if ( *(_DWORD *)(v5 + 36) != 210 )
    goto LABEL_5;
  v10 = *((_DWORD *)a3 + 1) & 0x7FFFFFF;
  v11 = *(_QWORD *)&a3[-32 * v10];
  if ( *(_BYTE *)v11 != 17 )
    goto LABEL_5;
  v12 = *(_DWORD *)(v11 + 32);
  if ( v12 <= 0x40 )
  {
    v13 = *(_QWORD *)(v11 + 24);
    goto LABEL_16;
  }
  if ( v12 - (unsigned int)sub_C444A0(v11 + 24) > 0x40 )
  {
LABEL_5:
    v6 = sub_D5D560((__int64)a3, *(__int64 **)(a2 + 808));
    if ( v6 )
    {
      *(_QWORD *)a1 = v6;
      *(_QWORD *)(a1 + 8) = 0xBFFFFFFFFFFFFFFELL;
      *(_QWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 24) = 0;
      *(_QWORD *)(a1 + 32) = 0;
      *(_QWORD *)(a1 + 40) = 0;
      *(_BYTE *)(a1 + 48) = 1;
      *(_BYTE *)(a1 + 56) = 1;
      return a1;
    }
LABEL_9:
    *(_BYTE *)(a1 + 56) = 0;
    return a1;
  }
  v13 = **(_QWORD **)(v11 + 24);
LABEL_16:
  v14 = *(_QWORD *)&a3[32 * (1 - v10)];
  if ( !v14 )
    goto LABEL_5;
  *(_QWORD *)a1 = v14;
  *(_BYTE *)(a1 + 48) = 0;
  if ( v13 > 0x3FFFFFFFFFFFFFFBLL )
    v13 = 0xBFFFFFFFFFFFFFFELL;
  *(_BYTE *)(a1 + 56) = 1;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 8) = v13;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  return a1;
}
