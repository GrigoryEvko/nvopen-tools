// Function: sub_852780
// Address: 0x852780
//
void __fastcall sub_852780(__int64 a1, __int64 *a2)
{
  __int64 v2; // rsi
  __int64 v3; // r12
  unsigned __int8 *v4; // rdi
  __int64 v5; // rax
  char *v6; // rdi
  char *v7; // rax
  _QWORD *v8; // rbx
  __int64 v9; // rbx
  __int64 v10; // r15
  int v11; // ebx
  signed int v12; // ebx
  char *v13; // r14
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 off; // [rsp+8h] [rbp-48h]
  int ptr; // [rsp+14h] [rbp-3Ch] BYREF
  __time_t v18[7]; // [rsp+18h] [rbp-38h] BYREF

  if ( unk_4F04D84
    && !(dword_4F04C64 | dword_4D03BB0[0] | unk_4D03CA8)
    && unk_4D03CD8 == -1
    && !(qword_4F074B0 | qword_4D03BD8)
    && (*(_BYTE *)(qword_4F04C68[0] + 9LL) & 0x10) == 0
    && dword_4F066AC
    && *(_QWORD *)(qword_4F07280 + 40LL) )
  {
    if ( sub_7E16F0() )
      sub_8163B0(1, a2);
    v2 = dword_4D04730;
    if ( dword_4D04730 )
      sub_8B1870();
    sub_7302F0();
    if ( dword_4F077C4 == 2 )
    {
      v14 = (int)dword_4F04C40;
      v15 = 776LL * (int)dword_4F04C40;
      *(_BYTE *)(qword_4F04C68[0] + v15 + 7) &= ~8u;
      if ( *(_QWORD *)(qword_4F04C68[0] + v15 + 456) )
        sub_8845B0(v14);
    }
    if ( unk_4F04D84 )
    {
      if ( !(dword_4F04C64 | dword_4D03BB0[0] | unk_4D03CA8) && unk_4D03CD8 == -1 )
      {
        v3 = qword_4F074B0 | qword_4D03BD8;
        if ( !(qword_4F074B0 | qword_4D03BD8)
          && (*(_BYTE *)(qword_4F04C68[0] + 9LL) & 0x10) == 0
          && dword_4F066AC
          && *(_QWORD *)(qword_4F07280 + 40LL) )
        {
          ptr = 0;
          if ( dword_4D04504 )
          {
            v4 = (unsigned __int8 *)unk_4D044F0;
          }
          else
          {
            v2 = (__int64)".pch";
            v4 = (unsigned __int8 *)sub_722560(qword_4F076F0, ".pch");
          }
          qword_4F5FB50 = (__int64)sub_851FD0(v4, v2);
          if ( (unsigned int)sub_7244C0(qword_4F5FB50) )
            sub_7212E0(qword_4F5FB50);
          qword_4F5FB40 = (FILE *)sub_685E40((char *)qword_4F5FB50, 1, 0, 0, 1698);
          v5 = sub_723260((char *)qword_4F5FB50);
          sub_67EA10(633, v5);
          v6 = byte_4F5FB80;
          if ( fwrite(byte_4F5FB80, size, 1u, qword_4F5FB40) != 1 )
            goto LABEL_49;
          off = ftell(qword_4F5FB40);
          fwrite(&ptr, 4u, 1u, qword_4F5FB40);
          sub_851CB0(qword_4F076B0);
          v7 = sub_722430(qword_4F076F0, 0);
          sub_851CB0(v7);
          sub_851D50((__int64 *)qword_4F5FB60);
          sub_851D50((__int64 *)qword_4F5FB70);
          sub_851E50(qword_4F07280);
          v8 = (_QWORD *)qword_4F07320[0];
          if ( qword_4F07320[0] )
          {
            do
            {
              sub_723E40(*(_QWORD *)(v8[4] + 24LL), v18);
              sub_851CB0(*(const char **)(v8[4] + 24LL));
              fwrite(v18, 8u, 1u, qword_4F5FB40);
              v8 = (_QWORD *)*v8;
            }
            while ( v8 );
          }
          v18[0] = 0;
          fwrite(v18, 8u, 1u, qword_4F5FB40);
          fwrite(&qword_4F07390, 8u, 1u, qword_4F5FB40);
          fwrite(&qword_4F07388, 8u, 1u, qword_4F5FB40);
          v6 = (char *)qword_4F07380;
          if ( fwrite(qword_4F07380, 16LL * qword_4F07388, 1u, qword_4F5FB40) != 1 )
            goto LABEL_49;
          if ( dword_4F5F92C > 0 )
          {
            v9 = 0;
            while ( 1 )
            {
              v10 = qword_4F5F940[v9];
              v6 = *(char **)v10;
              if ( *(_QWORD *)v10 )
                break;
LABEL_36:
              if ( dword_4F5F92C <= (int)++v9 )
                goto LABEL_37;
            }
            while ( 1 )
            {
              if ( *(_BYTE *)(v10 + 16) )
                v6 = *(char **)v6;
              if ( fwrite(v6, *(_QWORD *)(v10 + 8), 1u, qword_4F5FB40) != 1 )
                break;
              v6 = *(char **)(v10 + 24);
              v10 += 24;
              if ( !v6 )
                goto LABEL_36;
            }
LABEL_49:
            sub_851C80(v6);
          }
LABEL_37:
          fwrite(&qword_4F07280, 0xA8u, 1u, qword_4F5FB40);
          v11 = dword_4F073A8 + 1;
          fwrite(&dword_4F073A8, 4u, 1u, qword_4F5FB40);
          v6 = (char *)qword_4F073B0;
          if ( fwrite(qword_4F073B0, 8LL * v11, 1u, qword_4F5FB40) != 1 )
            goto LABEL_49;
          v6 = (char *)qword_4F072B0;
          if ( fwrite(qword_4F072B0, 8LL * v11, 1u, qword_4F5FB40) != 1 )
            goto LABEL_49;
          v12 = dword_4F073A0 + 1;
          fwrite(&dword_4F073A0, 4u, 1u, qword_4F5FB40);
          if ( v12 > 1 )
          {
            v6 = (char *)qword_4F072B8;
            if ( fwrite(qword_4F072B8, 16LL * v12, 1u, qword_4F5FB40) != 1 )
              goto LABEL_49;
          }
          if ( qword_4F07398 > 0 )
          {
            do
            {
              v13 = (char *)qword_4F07380 + 16 * v3;
              sub_721A70(qword_4F5FB40);
              v6 = *(char **)v13;
              if ( fwrite(*(const void **)v13, *((_QWORD *)v13 + 1), 1u, qword_4F5FB40) != 1 )
                goto LABEL_49;
            }
            while ( qword_4F07398 > ++v3 );
          }
          v6 = (char *)qword_4F5FB40;
          if ( fseek(qword_4F5FB40, off, 0) )
            goto LABEL_49;
          ptr = 1;
          fwrite(&ptr, 4u, 1u, qword_4F5FB40);
          fclose(qword_4F5FB40);
          qword_4F5FB40 = 0;
        }
      }
    }
  }
}
