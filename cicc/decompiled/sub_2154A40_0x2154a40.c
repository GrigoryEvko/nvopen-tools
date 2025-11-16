// Function: sub_2154A40
// Address: 0x2154a40
//
_QWORD *__fastcall sub_2154A40(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v6; // rbx
  __int64 v7; // r14
  __int64 v8; // rax
  __int64 v9; // rdx
  _BYTE *v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rdx
  _BYTE *v13; // rsi
  _QWORD v15[7]; // [rsp+8h] [rbp-38h] BYREF

  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  if ( a3 )
  {
    v4 = *(unsigned int *)(a3 + 8);
    if ( (unsigned int)v4 > 1 )
    {
      v6 = *(unsigned int *)(a3 + 8);
      v7 = 1;
      while ( 1 )
      {
        v8 = *(_QWORD *)(a3 + 8 * (v7 - v4));
        v9 = *(unsigned int *)(v8 + 8);
        v15[0] = v8;
        v10 = *(_BYTE **)(v8 - 8 * v9);
        if ( *v10 )
          goto LABEL_10;
        v11 = sub_161E970((__int64)v10);
        if ( v12 == 24 )
        {
          if ( *(_QWORD *)v11 ^ 0x656E72656B5F6C63LL | *(_QWORD *)(v11 + 8) ^ 0x64615F6772615F6CLL
            || *(_QWORD *)(v11 + 16) != 0x65636170735F7264LL )
          {
            goto LABEL_10;
          }
          goto LABEL_14;
        }
        if ( v12 == 25 )
          break;
        if ( v12 != 18 )
        {
          if ( v12 != 23
            || *(_QWORD *)v11 ^ 0x656E72656B5F6C63LL | *(_QWORD *)(v11 + 8) ^ 0x79745F6772615F6CLL
            || *(_DWORD *)(v11 + 16) != 1902077296
            || *(_WORD *)(v11 + 20) != 24949
            || *(_BYTE *)(v11 + 22) != 108 )
          {
            goto LABEL_10;
          }
LABEL_14:
          v13 = (_BYTE *)a1[1];
          if ( v13 == (_BYTE *)a1[2] )
            goto LABEL_27;
          goto LABEL_15;
        }
        if ( !(*(_QWORD *)v11 ^ 0x656E72656B5F6C63LL | *(_QWORD *)(v11 + 8) ^ 0x79745F6772615F6CLL)
          && *(_WORD *)(v11 + 16) == 25968 )
        {
          goto LABEL_14;
        }
        if ( *(_QWORD *)v11 ^ 0x656E72656B5F6C63LL | *(_QWORD *)(v11 + 8) ^ 0x616E5F6772615F6CLL
          || *(_WORD *)(v11 + 16) != 25965 )
        {
          goto LABEL_10;
        }
        v13 = (_BYTE *)a1[1];
        if ( v13 == (_BYTE *)a1[2] )
        {
LABEL_27:
          sub_21546F0((__int64)a1, v13, v15);
LABEL_10:
          if ( ++v7 == v6 )
            return a1;
          goto LABEL_11;
        }
LABEL_15:
        if ( v13 )
        {
          *(_QWORD *)v13 = v15[0];
          v13 = (_BYTE *)a1[1];
        }
        ++v7;
        a1[1] = v13 + 8;
        if ( v7 == v6 )
          return a1;
LABEL_11:
        v4 = *(unsigned int *)(a3 + 8);
      }
      if ( *(_QWORD *)v11 ^ 0x656E72656B5F6C63LL | *(_QWORD *)(v11 + 8) ^ 0x63615F6772615F6CLL
        || *(_QWORD *)(v11 + 16) != 0x6175715F73736563LL
        || *(_BYTE *)(v11 + 24) != 108 )
      {
        goto LABEL_10;
      }
      goto LABEL_14;
    }
  }
  return a1;
}
