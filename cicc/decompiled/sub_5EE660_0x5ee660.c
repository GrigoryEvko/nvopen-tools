// Function: sub_5EE660
// Address: 0x5ee660
//
void __fastcall sub_5EE660(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // rbx
  int v4; // r12d
  __int64 v5; // r13
  char v6; // al
  __int64 v7; // r15
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 *v10; // rax
  int v11[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v2 = *(_QWORD *)(a1 + 88);
  if ( *(_BYTE *)(v2 + 80) != 20 )
  {
    v3 = *(_QWORD *)(v2 + 8);
    v4 = (*(_BYTE *)(v2 + 81) & 0x10) == 0 ? 24 : 16;
    if ( v3 )
    {
      v5 = *(_QWORD *)(a1 + 88);
      while ( 1 )
      {
        v6 = *(_BYTE *)(v3 + 80);
        if ( v6 == (_BYTE)v4 )
        {
          v7 = v3;
          if ( (_BYTE)v4 == 16 )
          {
            v7 = **(_QWORD **)(v3 + 88);
            v6 = *(_BYTE *)(v7 + 80);
          }
          if ( v6 == 24 )
          {
            v7 = *(_QWORD *)(v7 + 88);
            v6 = *(_BYTE *)(v7 + 80);
          }
          if ( v6 != 20 && (v6 != 2 || (v8 = *(_QWORD *)(v7 + 88)) == 0 || *(_BYTE *)(v8 + 173) != 12) )
          {
            if ( (unsigned int)sub_5E9110(v2, v7, v11) )
              break;
          }
        }
        v5 = v3;
        v3 = *(_QWORD *)(v3 + 8);
LABEL_5:
        if ( !v3 )
          return;
      }
      v9 = dword_4F077BC;
      if ( dword_4F077BC )
      {
        v9 = qword_4F077A8 > 0x76BFu;
        if ( !v11[0] )
          goto LABEL_18;
      }
      else if ( !v11[0] )
      {
        goto LABEL_18;
      }
      if ( !(unsigned int)sub_7D0550(v2, v7, v9, 0) )
        sub_686C60(734, a2, v2, v7);
LABEL_18:
      v3 = *(_QWORD *)(v3 + 8);
      *(_QWORD *)(v5 + 8) = v3;
      if ( (*(_BYTE *)(v2 + 81) & 0x10) != 0 )
      {
        v10 = *(__int64 **)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v2 + 64) + 168LL) + 152LL) + 176LL);
        if ( v10 )
        {
          while ( *(_QWORD *)(v7 + 88) != v10[3] )
          {
            v10 = (__int64 *)*v10;
            if ( !v10 )
              goto LABEL_5;
          }
          *((_BYTE *)v10 + 40) |= 8u;
          v3 = *(_QWORD *)(v5 + 8);
        }
      }
      goto LABEL_5;
    }
  }
}
