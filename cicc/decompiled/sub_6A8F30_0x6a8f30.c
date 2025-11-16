// Function: sub_6A8F30
// Address: 0x6a8f30
//
__int64 __fastcall sub_6A8F30(_QWORD *a1, FILE *a2)
{
  __int64 v3; // rax
  char i; // dl
  __int64 v5; // r12
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  bool v10; // zf
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rdi
  __int64 v17; // rax
  __int64 v18; // rcx
  __int64 IO_read_ptr_low; // rdx
  __int64 v20; // rax
  __int64 v21; // [rsp+0h] [rbp-190h] BYREF
  FILE v22; // [rsp+8h] [rbp-188h] BYREF

  if ( !a1 )
  {
    ++*(_BYTE *)(qword_4F061C8 + 75LL);
    if ( (_DWORD)a2 != 2 )
    {
      if ( (unsigned int)a2 <= 2 )
      {
        if ( (_DWORD)a2 == 1 )
        {
          *(_QWORD *)&v22._flags = *(_QWORD *)&dword_4F063F8;
          sub_65CD60(&v21);
LABEL_6:
          v3 = v21;
          for ( i = *(_BYTE *)(v21 + 140); i == 12; i = *(_BYTE *)(v3 + 140) )
            v3 = *(_QWORD *)(v3 + 160);
          if ( i )
          {
            v5 = sub_726700(22);
            *(_QWORD *)v5 = sub_72CBE0(22, a2, v6, v7, v8, v9);
            *(_QWORD *)(v5 + 56) = v21;
          }
          else
          {
            v5 = sub_726700(0);
          }
          v10 = *(_BYTE *)(v5 + 24) == 22;
          *(_QWORD *)(v5 + 28) = *(_QWORD *)&v22._flags;
          if ( v10 && (*(_BYTE *)(qword_4D03C50 + 19LL) & 2) != 0 )
          {
LABEL_12:
            sub_6E70E0(v5, &v22._IO_read_ptr);
            HIDWORD(v22._IO_save_base) = v22._flags;
            LOWORD(v22._IO_backup_base) = *((_WORD *)&v22._flags + 2);
            *(_QWORD *)dword_4F07508 = *(char **)((char *)&v22._IO_save_base + 4);
            *(char **)((char *)&v22._IO_backup_base + 4) = *(char **)&dword_4F077C8;
            unk_4F061D8 = *(_QWORD *)&dword_4F077C8;
            sub_6E3280(&v22._IO_read_ptr, &dword_4F077C8);
            sub_6F6F40(&v22._IO_read_ptr, 0);
            goto LABEL_17;
          }
          goto LABEL_17;
        }
LABEL_47:
        sub_721090(a1);
      }
      sub_69ED20((__int64)&v22._IO_read_ptr, 0, 18, 0);
LABEL_14:
      if ( (_DWORD)a2 == 4 )
        sub_6F69D0(&v22._IO_read_ptr, 0);
      v5 = sub_6F6F40(&v22._IO_read_ptr, 0);
      goto LABEL_17;
    }
    v21 = 0;
    *(_QWORD *)&v22._flags = *(_QWORD *)&dword_4F063F8;
    if ( dword_4F077C4 == 2 )
    {
      if ( word_4F06418[0] == 1 && (unk_4D04A11 & 2) != 0 )
        goto LABEL_34;
      a2 = 0;
      if ( (unsigned int)sub_7C0F00(4096, 0) )
        goto LABEL_34;
    }
    else if ( word_4F06418[0] == 1 )
    {
LABEL_34:
      a2 = 0;
      v16 = 4096;
      LODWORD(v22._IO_read_ptr) = 0;
      v17 = sub_7BF130(4096, 0, &v22._IO_read_ptr);
      IO_read_ptr_low = LODWORD(v22._IO_read_ptr);
      if ( LODWORD(v22._IO_read_ptr) )
      {
        v16 = 40;
        sub_6851D0(0x28u);
      }
      else if ( *(_BYTE *)(v17 + 80) == 19 )
      {
        v20 = *(_QWORD *)(v17 + 88);
        if ( (*(_BYTE *)(v20 + 266) & 1) != 0 )
          v20 = *(_QWORD *)(*(_QWORD *)(v20 + 200) + 88LL);
        v21 = *(_QWORD *)(v20 + 104);
      }
      else
      {
        a2 = &v22;
        v16 = 730;
        sub_6854C0(0x2DAu, &v22, v17);
      }
      sub_7B8B50(v16, a2, IO_read_ptr_low, v18);
      goto LABEL_27;
    }
    sub_6851D0(0x28u);
    goto LABEL_27;
  }
  if ( (_DWORD)a2 != 2 )
  {
    if ( (unsigned int)a2 <= 2 )
    {
      if ( (_DWORD)a2 == 1 )
      {
        a2 = (FILE *)&v21;
        sub_6E44B0(a1, &v21, &v22);
        goto LABEL_6;
      }
      goto LABEL_47;
    }
    sub_6F8800(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*a1 + 64LL) + 16LL) + 16LL), a1, &v22._IO_read_ptr);
    goto LABEL_14;
  }
  a2 = (FILE *)&v21;
  v21 = 0;
  sub_6E4510(a1, &v21, &v22);
LABEL_27:
  if ( v21 )
  {
    v5 = sub_726700(37);
    *(_QWORD *)v5 = sub_72CBE0(37, a2, v12, v13, v14, v15);
    *(_QWORD *)(v5 + 56) = v21;
    *(_QWORD *)(v5 + 28) = *(_QWORD *)&v22._flags;
    if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 2) == 0 || !*(_BYTE *)(v5 + 24) )
      goto LABEL_17;
    goto LABEL_12;
  }
  v5 = sub_726700(0);
LABEL_17:
  if ( a1 )
    a1[2] = *(_QWORD *)(a1[2] + 16LL);
  else
    --*(_BYTE *)(qword_4F061C8 + 75LL);
  return v5;
}
