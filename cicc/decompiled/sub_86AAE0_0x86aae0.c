// Function: sub_86AAE0
// Address: 0x86aae0
//
void __fastcall sub_86AAE0(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // r13
  __int64 *v3; // rax
  _QWORD *v4; // r8
  __int64 *v5; // rsi
  __int64 *v6; // rax
  __int64 *v7; // rdx
  char v8; // cl
  __int64 **v9; // rcx
  __int64 v10; // rcx
  __int64 v11; // rcx
  int v12[9]; // [rsp+Ch] [rbp-24h] BYREF

  v1 = *(_QWORD *)(a1 + 32);
  v2 = *(_QWORD *)(v1 + 96);
  if ( v2 && *(_BYTE *)(v2 + 16) != 53 )
  {
    sub_7296C0(v12);
    if ( (*(_BYTE *)(v1 + 203) & 1) == 0
      || (*(_BYTE *)(v1 + 89) & 4) != 0 && (*(_BYTE *)(v1 + 195) & 2) != 0 && !*(_QWORD *)(v1 + 240) )
    {
      sub_86AA40(v1);
      if ( dword_4F077C4 != 2 )
        goto LABEL_13;
    }
    else
    {
      sub_86A080((_QWORD *)v2);
      v3 = *(__int64 **)(*(_QWORD *)(*(_QWORD *)(v1 + 40) + 32LL) + 96LL);
      do
        v3 = (__int64 *)*v3;
      while ( v3 && (*((_BYTE *)v3 + 16) != 53 || v1 != *(_QWORD *)(v3[3] + 24)) );
      *(_QWORD *)(v1 + 96) = v3;
      if ( dword_4F077C4 != 2 )
        goto LABEL_13;
    }
    v4 = *(_QWORD **)(a1 + 264);
    if ( v4 )
    {
      v5 = *(__int64 **)(*(_QWORD *)(a1 + 32) + 96LL);
      while ( 1 )
      {
        v6 = (__int64 *)v4[1];
        if ( v6 )
          break;
LABEL_31:
        v4 = (_QWORD *)*v4;
        if ( !v4 )
          goto LABEL_13;
      }
      while ( 1 )
      {
        v7 = v6;
        v6 = (__int64 *)*v6;
        v8 = *((_BYTE *)v7 + 16);
        if ( v8 == 58 )
        {
          if ( (*(_BYTE *)(*(_QWORD *)(v7[3] + 24) - 8LL) & 1) != 0 )
          {
            v9 = (__int64 **)v7[1];
            *((_BYTE *)v7 - 8) |= 0x80u;
            if ( !v9 )
              goto LABEL_36;
            goto LABEL_25;
          }
LABEL_21:
          if ( !v6 )
            goto LABEL_31;
        }
        else
        {
          if ( dword_4F077C4 != 2 )
            goto LABEL_21;
          if ( v8 != 53 )
            goto LABEL_21;
          v11 = v7[3];
          if ( (*(_BYTE *)(v11 + 57) & 8) == 0 || *(char *)(*(_QWORD *)(v11 + 24) - 8LL) >= 0 )
            goto LABEL_21;
          *((_BYTE *)v7 - 8) |= 0x80u;
          *(_BYTE *)(v11 - 8) |= 0x80u;
          v9 = (__int64 **)v7[1];
          if ( !v9 )
          {
LABEL_36:
            v4[1] = v6;
            goto LABEL_26;
          }
LABEL_25:
          *v9 = v6;
LABEL_26:
          if ( v6 )
            v6[1] = v7[1];
          v10 = *v5;
          *v7 = *v5;
          if ( v10 )
          {
            *(_QWORD *)(v10 + 8) = v7;
          }
          else if ( !*(_QWORD *)(*(_QWORD *)(qword_4F04C68[0] + 184LL) + 256LL) )
          {
            *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C34 + 336) = v7;
          }
          *v5 = (__int64)v7;
          v7[1] = (__int64)v5;
          v5 = v7;
          if ( !v6 )
            goto LABEL_31;
        }
      }
    }
LABEL_13:
    sub_729730(v12[0]);
  }
}
