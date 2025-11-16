// Function: sub_B17DD0
// Address: 0xb17dd0
//
_BYTE *__fastcall sub_B17DD0(__int64 a1)
{
  _BYTE *result; // rax
  _BYTE *v2; // rbx
  int v3; // r14d
  const char *v4; // r15
  size_t v5; // r12
  __int64 v6; // rax
  bool v7; // zf
  __int64 v8; // rax
  unsigned __int8 v9; // dl
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // r15
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r12
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // [rsp+8h] [rbp-88h]
  __int64 v20; // [rsp+18h] [rbp-78h] BYREF
  void *v21; // [rsp+20h] [rbp-70h] BYREF
  int v22; // [rsp+28h] [rbp-68h]
  char v23; // [rsp+2Ch] [rbp-64h]
  __int64 v24; // [rsp+30h] [rbp-60h]
  __int64 v25; // [rsp+38h] [rbp-58h]
  __int64 v26; // [rsp+40h] [rbp-50h]
  __int64 v27; // [rsp+48h] [rbp-48h]
  __int64 v28; // [rsp+50h] [rbp-40h]

  result = (_BYTE *)sub_BD3990(*(_QWORD *)(a1 - 32));
  if ( !*result )
  {
    v2 = result;
    v3 = 0;
    v4 = "dontcall-error";
    while ( 1 )
    {
      v5 = strlen(v4);
      result = (_BYTE *)sub_B2D620(v2, v4, v5);
      if ( (_BYTE)result )
      {
        v6 = sub_B2D7E0(v2, v4, v5);
        v7 = *(_QWORD *)(a1 + 48) == 0;
        v20 = v6;
        if ( !v7 || (v12 = 0, (*(_BYTE *)(a1 + 7) & 0x20) != 0) )
        {
          v8 = sub_B91F50(a1, "srcloc", 6);
          if ( v8 )
          {
            v9 = *(_BYTE *)(v8 - 16);
            if ( (v9 & 2) != 0 )
              v10 = *(_QWORD *)(v8 - 32);
            else
              v10 = v8 - 8LL * ((v9 >> 2) & 0xF) - 16;
            v11 = *(_QWORD *)(*(_QWORD *)v10 + 136LL);
            v12 = *(_QWORD *)(v11 + 24);
            if ( *(_DWORD *)(v11 + 32) > 0x40u )
              v12 = **(_QWORD **)(v11 + 24);
          }
          else
          {
            v12 = 0;
          }
        }
        v13 = sub_A72240(&v20);
        v15 = v14;
        v19 = v13;
        v16 = sub_BD5D20(v2);
        v22 = 26;
        v23 = v3;
        v24 = v16;
        v21 = &unk_49D9FB8;
        v25 = v17;
        v26 = v19;
        v27 = v15;
        v28 = v12;
        v18 = sub_B2BE50(v2);
        result = (_BYTE *)sub_B6EB20(v18, &v21);
      }
      v4 = "dontcall-warn";
      if ( v3 == 1 )
        break;
      v3 = 1;
    }
  }
  return result;
}
