// Function: sub_30B3260
// Address: 0x30b3260
//
__int64 *__fastcall sub_30B3260(__int64 *a1, __int64 a2)
{
  int v3; // eax
  __int64 *v4; // rcx
  __int64 *v5; // rbx
  __int64 *v6; // r13
  unsigned __int64 *v7; // rax
  char *v9; // rdx
  char *v10; // rdx
  _QWORD *v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rdx
  unsigned __int64 v14[2]; // [rsp+10h] [rbp-90h] BYREF
  _BYTE v15[16]; // [rsp+20h] [rbp-80h] BYREF
  _QWORD v16[3]; // [rsp+30h] [rbp-70h] BYREF
  char *v17; // [rsp+48h] [rbp-58h]
  char *v18; // [rsp+50h] [rbp-50h]
  __int64 v19; // [rsp+58h] [rbp-48h]
  unsigned __int64 *v20; // [rsp+60h] [rbp-40h]

  v14[0] = (unsigned __int64)v15;
  v19 = 0x100000000LL;
  v20 = v14;
  v14[1] = 0;
  v16[0] = &unk_49DD210;
  v15[0] = 0;
  v16[1] = 0;
  v16[2] = 0;
  v17 = 0;
  v18 = 0;
  sub_CB5980((__int64)v16, 0, 0, 0);
  v3 = *(_DWORD *)(a2 + 56);
  if ( (unsigned int)(v3 - 1) > 1 )
  {
    if ( v3 == 3 )
    {
      v10 = v18;
      if ( (unsigned __int64)(v17 - v18) <= 0xD )
      {
        v11 = (_QWORD *)sub_CB6200((__int64)v16, "pi-block\nwith\n", 0xEu);
      }
      else
      {
        *((_DWORD *)v18 + 2) = 1953068810;
        v11 = v16;
        *(_QWORD *)v10 = 0x6B636F6C622D6970LL;
        *((_WORD *)v10 + 6) = 2664;
        v18 += 14;
      }
      v12 = sub_CB59D0((__int64)v11, *(unsigned int *)(a2 + 72));
      v13 = *(_QWORD *)(v12 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(v12 + 24) - v13) <= 6 )
      {
        sub_CB6200(v12, " nodes\n", 7u);
      }
      else
      {
        *(_DWORD *)v13 = 1685024288;
        *(_WORD *)(v13 + 4) = 29541;
        *(_BYTE *)(v13 + 6) = 10;
        *(_QWORD *)(v12 + 32) += 7LL;
      }
    }
    else
    {
      if ( v3 != 4 )
        BUG();
      v9 = v18;
      if ( (unsigned __int64)(v17 - v18) <= 4 )
      {
        sub_CB6200((__int64)v16, "root\n", 5u);
      }
      else
      {
        *(_DWORD *)v18 = 1953460082;
        v9[4] = 10;
        v18 += 5;
      }
    }
  }
  else
  {
    v4 = *(__int64 **)(a2 + 64);
    v5 = &v4[*(unsigned int *)(a2 + 72)];
    if ( v5 != v4 )
    {
      v6 = *(__int64 **)(a2 + 64);
      do
      {
        while ( 1 )
        {
          sub_A69870(*v6, v16, 0);
          if ( v17 == v18 )
            break;
          ++v6;
          *v18++ = 10;
          if ( v5 == v6 )
            goto LABEL_7;
        }
        ++v6;
        sub_CB6200((__int64)v16, (unsigned __int8 *)"\n", 1u);
      }
      while ( v5 != v6 );
    }
  }
LABEL_7:
  v7 = v20;
  *a1 = (__int64)(a1 + 2);
  sub_30B3180(a1, (_BYTE *)*v7, *v7 + v7[1]);
  v16[0] = &unk_49DD210;
  sub_CB5840((__int64)v16);
  if ( (_BYTE *)v14[0] != v15 )
    j_j___libc_free_0(v14[0]);
  return a1;
}
