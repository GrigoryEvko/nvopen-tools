// Function: sub_28B5F90
// Address: 0x28b5f90
//
void __fastcall sub_28B5F90(char *a1, char *a2)
{
  __int64 v2; // r15
  unsigned __int64 *v3; // r12
  __int64 v4; // rdi
  unsigned __int64 v5; // rax
  __int64 v6; // rdx
  unsigned int v7; // ecx
  unsigned int v8; // edx
  __int64 v9; // rcx
  unsigned __int64 v10; // r14
  unsigned __int64 v11; // rbx
  unsigned __int64 v12; // r13
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // r15
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // r12
  __int64 v20; // rbx
  unsigned __int64 v21; // r14
  unsigned __int64 v22; // rdi
  unsigned __int64 v23; // rdi
  __int64 v24; // [rsp+0h] [rbp-70h]
  __int64 v25; // [rsp+8h] [rbp-68h]
  __int64 v26; // [rsp+18h] [rbp-58h]
  char *v27; // [rsp+20h] [rbp-50h]
  char *v29; // [rsp+30h] [rbp-40h]

  if ( a1 != a2 && a2 != a1 + 24 )
  {
    v29 = a1 + 48;
    do
    {
      v2 = *((_QWORD *)v29 - 2);
      v27 = v29;
      v3 = (unsigned __int64 *)(v29 - 24);
      v26 = *((_QWORD *)v29 - 3);
      v4 = *((_QWORD *)a1 + 1);
      v5 = *(_QWORD *)a1;
      if ( v26 == v2 )
      {
        v7 = -1;
        if ( v5 == v4 )
          goto LABEL_47;
      }
      else
      {
        v6 = *((_QWORD *)v29 - 3);
        v7 = -1;
        do
        {
          if ( v7 > *(_DWORD *)(v6 + 92) )
            v7 = *(_DWORD *)(v6 + 92);
          v6 += 192;
        }
        while ( v2 != v6 );
        if ( v5 == v4 )
        {
          v8 = -1;
          goto LABEL_14;
        }
      }
      v8 = -1;
      do
      {
        if ( v8 > *(_DWORD *)(v5 + 92) )
          v8 = *(_DWORD *)(v5 + 92);
        v5 += 192LL;
      }
      while ( v4 != v5 );
LABEL_14:
      if ( v8 > v7 )
      {
        v9 = *((_QWORD *)v29 - 1);
        *((_QWORD *)v29 - 2) = 0;
        *((_QWORD *)v29 - 1) = 0;
        *((_QWORD *)v29 - 3) = 0;
        v25 = v9;
        v10 = 0xAAAAAAAAAAAAAAABLL * (((char *)v3 - a1) >> 3);
        if ( (char *)v3 - a1 > 0 )
        {
          v11 = 0;
          v12 = 0;
          v24 = v2;
          while ( 1 )
          {
            v13 = *(v3 - 3);
            v3 -= 3;
            v14 = v12;
            *v3 = 0;
            v3[3] = v13;
            v15 = v3[1];
            v3[1] = 0;
            v3[4] = v15;
            v16 = v3[2];
            v3[2] = 0;
            for ( v3[5] = v16; v11 != v14; v14 += 192LL )
            {
              if ( *(_DWORD *)(v14 + 168) > 0x40u )
              {
                v17 = *(_QWORD *)(v14 + 160);
                if ( v17 )
                  j_j___libc_free_0_0(v17);
              }
              if ( *(_DWORD *)(v14 + 128) > 0x40u )
              {
                v18 = *(_QWORD *)(v14 + 120);
                if ( v18 )
                  j_j___libc_free_0_0(v18);
              }
              if ( (*(_BYTE *)(v14 + 16) & 1) == 0 )
                sub_C7D6A0(*(_QWORD *)(v14 + 24), 8LL * *(unsigned int *)(v14 + 32), 8);
            }
            if ( v12 )
              j_j___libc_free_0(v12);
            if ( !--v10 )
              break;
            v12 = *v3;
            v11 = v3[1];
          }
          v2 = v24;
        }
        v19 = *(_QWORD *)a1;
        v20 = *((_QWORD *)a1 + 1);
        *(_QWORD *)a1 = v26;
        *((_QWORD *)a1 + 1) = v2;
        v21 = v19;
        for ( *((_QWORD *)a1 + 2) = v25; v20 != v21; v21 += 192LL )
        {
          if ( *(_DWORD *)(v21 + 168) > 0x40u )
          {
            v22 = *(_QWORD *)(v21 + 160);
            if ( v22 )
              j_j___libc_free_0_0(v22);
          }
          if ( *(_DWORD *)(v21 + 128) > 0x40u )
          {
            v23 = *(_QWORD *)(v21 + 120);
            if ( v23 )
              j_j___libc_free_0_0(v23);
          }
          if ( (*(_BYTE *)(v21 + 16) & 1) == 0 )
            sub_C7D6A0(*(_QWORD *)(v21 + 24), 8LL * *(unsigned int *)(v21 + 32), 8);
        }
        if ( v19 )
          j_j___libc_free_0(v19);
        goto LABEL_44;
      }
LABEL_47:
      sub_28B5D30((_QWORD *)v29 - 3);
LABEL_44:
      v29 += 24;
    }
    while ( a2 != v27 );
  }
}
