// Function: sub_D90530
// Address: 0xd90530
//
char __fastcall sub_D90530(__int64 *a1, __int64 a2)
{
  _QWORD *v2; // rax
  _QWORD *v3; // r12
  unsigned __int64 v4; // rbx
  _QWORD *v5; // r13
  _QWORD *v6; // rax
  _QWORD *v7; // rdi
  __int64 v8; // rcx
  __int64 v9; // rdx
  unsigned __int8 *v10; // rax
  _QWORD *v11; // rdi
  unsigned __int64 v12; // r13
  size_t v13; // rdx
  __int64 v14; // rax
  _QWORD *v15; // rbx
  _QWORD *i; // r14
  char v17; // al
  unsigned __int64 v18; // r11
  __int64 v19; // rax
  char v20; // al
  _QWORD *v21; // rax
  _QWORD *v23; // [rsp+0h] [rbp-60h]
  __int64 v24; // [rsp+8h] [rbp-58h]
  _QWORD *v26; // [rsp+20h] [rbp-40h]
  unsigned __int64 v27; // [rsp+28h] [rbp-38h]
  _BYTE *v28; // [rsp+28h] [rbp-38h]

  v2 = (_QWORD *)sub_D8E7E0(a1, a2);
  v23 = v2;
  if ( v2[5] )
  {
    v2 = *(_QWORD **)(*(_QWORD *)(v2[3] + 32LL) + 40LL);
    v3 = (_QWORD *)v2[4];
    v26 = v2 + 3;
    while ( v26 != v3 )
    {
      v4 = (unsigned __int64)(v3 - 7);
      if ( !v3 )
        v4 = 0;
      LOBYTE(v2) = sub_B2FC80(v4);
      if ( !(_BYTE)v2 )
      {
        v5 = v23 + 1;
        v6 = (_QWORD *)v23[2];
        if ( v6 )
        {
          v7 = v23 + 1;
          do
          {
            while ( 1 )
            {
              v8 = v6[2];
              v9 = v6[3];
              if ( v6[4] >= v4 )
                break;
              v6 = (_QWORD *)v6[3];
              if ( !v9 )
                goto LABEL_13;
            }
            v7 = v6;
            v6 = (_QWORD *)v6[2];
          }
          while ( v8 );
LABEL_13:
          if ( v5 != v7 && v7[4] <= v4 )
            v5 = v7;
        }
        v10 = (unsigned __int8 *)sub_BD5D20(v4);
        v11 = v5 + 5;
        v12 = v4 + 72;
        sub_D88690(v11, a2, v10, v13, v4);
        v14 = sub_904010(a2, "    safe accesses:");
        sub_904010(v14, "\n");
        v15 = *(_QWORD **)(v4 + 80);
        if ( (_QWORD *)v12 == v15 )
        {
          i = 0;
        }
        else
        {
          if ( !v15 )
            BUG();
          while ( 1 )
          {
            i = (_QWORD *)v15[4];
            if ( i != v15 + 3 )
              break;
            v15 = (_QWORD *)v15[1];
            if ( (_QWORD *)v12 == v15 )
              break;
            if ( !v15 )
              BUG();
          }
        }
LABEL_22:
        while ( (_QWORD *)v12 != v15 )
        {
          if ( !i )
            BUG();
          v17 = *((_BYTE *)i - 24);
          v18 = (unsigned __int64)(i - 3);
          if ( v17 == 85 )
          {
            v19 = *(i - 7);
            if ( !v19
              || *(_BYTE *)v19
              || *(_QWORD *)(v19 + 24) != i[7]
              || (*(_BYTE *)(v19 + 33) & 0x20) == 0
              || (unsigned int)(*(_DWORD *)(v19 + 36) - 238) > 7
              || ((1LL << (*(_BYTE *)(v19 + 36) + 18)) & 0xAD) == 0 )
            {
              v20 = sub_A74390(i + 6, 81, 0);
              v18 = (unsigned __int64)(i - 3);
              if ( !v20 )
                goto LABEL_37;
            }
          }
          else if ( (unsigned __int8)(v17 - 61) > 1u && (unsigned __int8)(v17 - 65) > 1u )
          {
            goto LABEL_37;
          }
          v27 = v18;
          if ( sub_D904B0(a1, v18) )
          {
            v24 = v27;
            v28 = (_BYTE *)sub_904010(a2, "     ");
            sub_A69870(v24, v28, 0);
            sub_904010((__int64)v28, "\n");
          }
LABEL_37:
          for ( i = (_QWORD *)i[1]; ; i = (_QWORD *)v15[4] )
          {
            v21 = v15 - 3;
            if ( !v15 )
              v21 = 0;
            if ( i != v21 + 6 )
              break;
            v15 = (_QWORD *)v15[1];
            if ( (_QWORD *)v12 == v15 )
              goto LABEL_22;
            if ( !v15 )
              BUG();
          }
        }
        LOBYTE(v2) = sub_904010(a2, "\n");
      }
      v3 = (_QWORD *)v3[1];
    }
  }
  return (char)v2;
}
