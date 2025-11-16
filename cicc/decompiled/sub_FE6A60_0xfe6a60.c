// Function: sub_FE6A60
// Address: 0xfe6a60
//
__int64 __fastcall sub_FE6A60(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  __int64 result; // rax
  __int64 v7; // r12
  __int64 v8; // r13
  int v9; // edx
  int v10; // ecx
  __int64 v11; // rdi
  unsigned int v12; // edx
  __int64 v13; // rsi
  unsigned __int64 v14; // r15
  int v15; // r8d
  __int64 v16; // rax
  __int64 v17; // [rsp+0h] [rbp-50h] BYREF
  __int64 v18; // [rsp+8h] [rbp-48h]
  __int64 v19; // [rsp+10h] [rbp-40h]

  *(_QWORD *)(a1 + 112) = a3;
  *(_QWORD *)(a1 + 120) = a4;
  *(_QWORD *)(a1 + 128) = a2;
  sub_FE9370();
  v5 = *(_QWORD *)(a1 + 136);
  if ( *(_QWORD *)(a1 + 144) != v5 )
    *(_QWORD *)(a1 + 144) = v5;
  sub_FDCC00(a1 + 160);
  sub_FE5320(a1);
  sub_FDF190((_QWORD *)a1);
  sub_FE69E0((_QWORD *)a1);
  if ( !(unsigned __int8)sub_FDE5F0((_QWORD *)a1) )
  {
    sub_FE67B0((_QWORD *)a1, 0, *(_QWORD *)(a1 + 88));
    if ( !(unsigned __int8)sub_FDE5F0((_QWORD *)a1) )
      BUG();
  }
  sub_FE9700(a1);
  if ( LOBYTE(qword_4F8E448[8]) )
  {
    sub_B2EE70((__int64)&v17, *(_QWORD *)(a1 + 128), 0);
    if ( (_BYTE)v19 )
    {
      v16 = a1 + 88;
      while ( *(_QWORD *)(a1 + 88) != v16 )
      {
        v16 = *(_QWORD *)(v16 + 8);
        if ( *(_DWORD *)(v16 + 28) > 1u )
        {
          sub_FE3DA0(a1);
          break;
        }
      }
    }
  }
  sub_FE98F0(a1);
  result = (__int64)&qword_4F8E4E0;
  if ( LOBYTE(qword_4F8E528[8]) )
  {
    v7 = *(_QWORD *)(a2 + 80);
    v8 = a2 + 72;
    if ( a2 + 72 != v7 )
    {
      while ( v7 )
      {
        v14 = v7 - 24;
        v17 = 0;
        v18 = 0;
        v19 = v7 - 24;
        if ( v7 == -4072 )
        {
          v9 = *(_DWORD *)(a1 + 184);
          result = -4096;
          if ( v9 )
            goto LABEL_10;
LABEL_19:
          result = sub_FE0A90(a1, v14, 0);
          v7 = *(_QWORD *)(v7 + 8);
          if ( v8 == v7 )
            return result;
        }
        else
        {
          if ( v7 == -8168 )
          {
            v9 = *(_DWORD *)(a1 + 184);
            result = -8192;
            if ( !v9 )
              goto LABEL_19;
          }
          else
          {
            sub_BD73F0((__int64)&v17);
            v9 = *(_DWORD *)(a1 + 184);
            result = v19;
            if ( !v9 )
              goto LABEL_27;
          }
LABEL_10:
          v10 = v9 - 1;
          v11 = *(_QWORD *)(a1 + 168);
          v12 = (v9 - 1) & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
          v13 = *(_QWORD *)(v11 + 72LL * v12 + 16);
          if ( v13 != result )
          {
            v15 = 1;
            while ( v13 != -4096 )
            {
              v12 = v10 & (v15 + v12);
              v13 = *(_QWORD *)(v11 + 72LL * v12 + 16);
              if ( v13 == result )
                goto LABEL_11;
              ++v15;
            }
LABEL_27:
            if ( result && result != -4096 && result != -8192 )
              sub_BD60C0(&v17);
            goto LABEL_19;
          }
LABEL_11:
          if ( result && result != -4096 && result != -8192 )
            result = sub_BD60C0(&v17);
          v7 = *(_QWORD *)(v7 + 8);
          if ( v8 == v7 )
            return result;
        }
      }
      v9 = *(_DWORD *)(a1 + 184);
      v14 = 0;
      result = 0;
      v17 = 0;
      v18 = 0;
      v19 = 0;
      if ( !v9 )
        goto LABEL_19;
      goto LABEL_10;
    }
  }
  return result;
}
