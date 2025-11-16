// Function: sub_17F4910
// Address: 0x17f4910
//
void __fastcall sub_17F4910(__int64 *a1, __int64 *a2)
{
  unsigned __int64 v2; // r15
  unsigned int v3; // ebx
  unsigned __int64 v4; // rax
  __int64 v5; // rcx
  __int64 v6; // r14
  unsigned int v7; // r12d
  __int64 v8; // r13
  int v9; // eax
  unsigned int v10; // ebx
  __int64 *i; // r15
  unsigned __int64 v12; // rax
  __int64 v13; // rbx
  unsigned __int64 v14; // r12
  unsigned int v15; // edx
  int v16; // eax
  unsigned int v17; // edx
  __int64 *v18; // [rsp+10h] [rbp-40h]
  __int64 v19; // [rsp+18h] [rbp-38h]
  __int64 v20; // [rsp+18h] [rbp-38h]
  int v21; // [rsp+18h] [rbp-38h]

  if ( a1 != a2 && a2 != a1 + 1 )
  {
    v18 = a1 + 1;
    while ( 1 )
    {
      v5 = *a1;
      v6 = *v18;
      v7 = *(_DWORD *)(*v18 + 32);
      v8 = *v18 + 24;
      if ( v7 <= 0x40 )
        break;
      v19 = *a1;
      v2 = -1;
      v9 = sub_16A57B0(*v18 + 24);
      v5 = v19;
      if ( v7 - v9 > 0x40 )
        goto LABEL_5;
      v3 = *(_DWORD *)(v19 + 32);
      v2 = **(_QWORD **)(v6 + 24);
      if ( v3 <= 0x40 )
      {
LABEL_6:
        v4 = *(_QWORD *)(v5 + 24);
        goto LABEL_7;
      }
LABEL_14:
      v20 = v5;
      v10 = v3 - sub_16A57B0(v5 + 24);
      v4 = -1;
      if ( v10 > 0x40 )
      {
LABEL_7:
        if ( v4 > v2 )
          goto LABEL_8;
LABEL_16:
        for ( i = v18; ; --i )
        {
          v13 = *(i - 1);
          if ( v7 > 0x40 )
          {
            v16 = sub_16A57B0(v8);
            v17 = v7;
            v14 = -1;
            if ( v17 - v16 <= 0x40 )
              v14 = **(_QWORD **)(v6 + 24);
          }
          else
          {
            v14 = *(_QWORD *)(v6 + 24);
          }
          if ( *(_DWORD *)(v13 + 32) <= 0x40u )
          {
            v12 = *(_QWORD *)(v13 + 24);
            goto LABEL_18;
          }
          v21 = *(_DWORD *)(v13 + 32);
          v15 = v21 - sub_16A57B0(v13 + 24);
          v12 = -1;
          if ( v15 <= 0x40 )
            break;
LABEL_18:
          if ( v12 <= v14 )
            goto LABEL_25;
LABEL_19:
          *i = v13;
          v7 = *(_DWORD *)(v6 + 32);
        }
        if ( **(_QWORD **)(v13 + 24) > v14 )
          goto LABEL_19;
LABEL_25:
        ++v18;
        *i = v6;
        if ( v18 == a2 )
          return;
      }
      else
      {
        if ( **(_QWORD **)(v20 + 24) <= v2 )
          goto LABEL_16;
LABEL_8:
        if ( a1 != v18 )
          memmove(a1 + 1, a1, (char *)v18 - (char *)a1);
        ++v18;
        *a1 = v6;
        if ( v18 == a2 )
          return;
      }
    }
    v2 = *(_QWORD *)(v6 + 24);
LABEL_5:
    v3 = *(_DWORD *)(v5 + 32);
    if ( v3 <= 0x40 )
      goto LABEL_6;
    goto LABEL_14;
  }
}
