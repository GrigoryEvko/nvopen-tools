// Function: sub_16B7FD0
// Address: 0x16b7fd0
//
void __fastcall sub_16B7FD0(__int64 a1, const char *a2, size_t a3)
{
  size_t v3; // r8
  __int64 v4; // r15
  __int64 v5; // rdx
  __int64 *v6; // rax
  __int64 *v7; // r12
  __int64 v8; // rdx
  __int64 *v9; // rbx
  __int64 *v10; // rax
  __int64 v11; // rax
  size_t v12; // [rsp+8h] [rbp-38h]
  size_t v13; // [rsp+8h] [rbp-38h]

  v3 = a3;
  if ( !qword_4FA01E0 )
  {
    sub_16C1EA0(&qword_4FA01E0, sub_16B89A0, sub_16B0D50);
    v3 = a3;
  }
  v4 = qword_4FA01E0;
  v5 = *(unsigned int *)(a1 + 108);
  if ( (_DWORD)v5 == *(_DWORD *)(a1 + 112) )
  {
    v13 = v3;
    v11 = sub_16B4B80((__int64)&unk_4FA0190);
    sub_16B7D30(v4, a1, v11, a2, v13);
  }
  else
  {
    v6 = *(__int64 **)(a1 + 96);
    if ( v6 != *(__int64 **)(a1 + 88) )
      v5 = *(unsigned int *)(a1 + 104);
    v7 = &v6[v5];
    if ( v6 != v7 )
    {
      while ( 1 )
      {
        v8 = *v6;
        v9 = v6;
        if ( (unsigned __int64)*v6 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v7 == ++v6 )
          return;
      }
      if ( v6 != v7 )
      {
        do
        {
          v12 = v3;
          sub_16B7D30(v4, a1, v8, a2, v3);
          v10 = v9 + 1;
          if ( v9 + 1 == v7 )
            break;
          v8 = *v10;
          v3 = v12;
          for ( ++v9; (unsigned __int64)*v10 >= 0xFFFFFFFFFFFFFFFELL; v9 = v10 )
          {
            if ( v7 == ++v10 )
              return;
            v8 = *v10;
          }
        }
        while ( v7 != v9 );
      }
    }
  }
}
