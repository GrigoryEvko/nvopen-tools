// Function: sub_1666000
// Address: 0x1666000
//
void __fastcall sub_1666000(__int64 a1, __int64 a2)
{
  __int64 v3; // r14
  unsigned __int64 v4; // rax
  __int64 v5; // rdx
  const char *v6; // rax
  __int64 v7; // r14
  _BYTE *v8; // rax
  __int64 v9; // rax
  const char *v10; // rax
  __int64 v11; // rdi
  _QWORD v12[2]; // [rsp+0h] [rbp-40h] BYREF
  char v13; // [rsp+10h] [rbp-30h]
  char v14; // [rsp+11h] [rbp-2Fh]

  if ( *(_BYTE *)(**(_QWORD **)(a2 - 24) + 8LL) == 15 )
  {
    if ( (unsigned int)(1 << (*(unsigned __int16 *)(a2 + 18) >> 1)) <= 0x40000001 )
    {
      v3 = *(_QWORD *)a2;
      v4 = *(unsigned __int8 *)(*(_QWORD *)a2 + 8LL);
      if ( (unsigned __int8)v4 > 0xFu || (v5 = 35454, !_bittest64(&v5, v4)) )
      {
        if ( (unsigned int)(v4 - 13) > 1 && (_DWORD)v4 != 16 || !sub_16435F0(v3, 0) )
        {
          v14 = 1;
          v10 = "loading unsized types is not allowed";
          goto LABEL_24;
        }
      }
      if ( sub_15F32D0(a2) )
      {
        if ( ((*(unsigned __int16 *)(a2 + 18) >> 7) & 7u) - 5 <= 1 )
        {
          v14 = 1;
          v10 = "Load cannot have Release ordering";
        }
        else
        {
          if ( (unsigned int)(1 << (*(unsigned __int16 *)(a2 + 18) >> 1)) >> 1 )
          {
            if ( (*(_BYTE *)(v3 + 8) & 0xFB) != 0xB && (unsigned __int8)(*(_BYTE *)(v3 + 8) - 1) > 5u )
            {
              v14 = 1;
              v12[0] = "atomic load operand must have integer, pointer, or floating point type!";
              v13 = 3;
              sub_164FF40((__int64 *)a1, (__int64)v12);
              v11 = *(_QWORD *)a1;
              if ( !*(_QWORD *)a1 )
                return;
              sub_164ECF0(v11, v3);
              goto LABEL_16;
            }
            sub_164FB00(a1, v3, a2);
LABEL_10:
            sub_1663F80(a1, a2);
            return;
          }
          v14 = 1;
          v10 = "Atomic load must specify explicit alignment";
        }
      }
      else
      {
        if ( *(_BYTE *)(a2 + 56) == 1 )
          goto LABEL_10;
        v14 = 1;
        v10 = "Non-atomic load cannot have SynchronizationScope specified";
      }
LABEL_24:
      v12[0] = v10;
      v13 = 3;
      sub_164FF40((__int64 *)a1, (__int64)v12);
      if ( !*(_QWORD *)a1 )
        return;
LABEL_16:
      sub_164FA80((__int64 *)a1, a2);
      return;
    }
    v14 = 1;
    v6 = "huge alignment values are unsupported";
  }
  else
  {
    v14 = 1;
    v6 = "Load operand must be a pointer.";
  }
  v7 = *(_QWORD *)a1;
  v12[0] = v6;
  v13 = 3;
  if ( !v7 )
  {
    *(_BYTE *)(a1 + 72) = 1;
    return;
  }
  sub_16E2CE0(v12, v7);
  v8 = *(_BYTE **)(v7 + 24);
  if ( (unsigned __int64)v8 >= *(_QWORD *)(v7 + 16) )
  {
    sub_16E7DE0(v7, 10);
  }
  else
  {
    *(_QWORD *)(v7 + 24) = v8 + 1;
    *v8 = 10;
  }
  v9 = *(_QWORD *)a1;
  *(_BYTE *)(a1 + 72) = 1;
  if ( v9 )
    goto LABEL_16;
}
