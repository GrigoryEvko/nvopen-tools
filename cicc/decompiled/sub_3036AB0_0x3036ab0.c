// Function: sub_3036AB0
// Address: 0x3036ab0
//
__int64 __fastcall sub_3036AB0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, int a5)
{
  __int64 v9; // rdi
  __int64 (*v10)(void); // rax
  __int64 v11; // r14
  __int64 result; // rax
  __int64 *v13; // r13
  __int64 v14; // rax
  __int64 *v15; // rdx
  __int64 v16; // rsi
  __int64 v17; // rdx
  bool v18; // zf
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rax
  unsigned __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rdx
  unsigned __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 (__fastcall *v38)(__int64, __int64, unsigned int); // rcx
  int v39; // edx
  __int16 v40; // ax
  __int64 v41; // rax
  __int64 v42; // rax
  bool v43; // cc
  unsigned __int64 v44; // rax
  char v45; // dl
  __int64 v46; // rax
  __int64 *v47; // r13
  __int64 v48; // rax
  int v49; // eax
  __int64 v50; // rdx
  __int64 *v51; // r13
  __int64 v52; // rax
  __int64 v53; // rdx
  __int64 *v54; // r13
  __int64 v55; // rax
  __int64 v56; // rdx
  __int64 *v57; // r13
  __int64 v58; // rax
  int v59; // eax
  int v60; // esi
  __int64 v61; // rdx
  __int16 v62; // ax
  __int64 *v63; // r15
  __int64 v64; // rax
  __int64 v65; // rdx
  __int64 v66; // rdx
  __int64 *v67; // r13
  __int64 v68; // rax
  __int64 v69; // rdx
  char v70; // [rsp+Fh] [rbp-31h]

  v9 = a1[67127];
  v10 = *(__int64 (**)(void))(*(_QWORD *)v9 + 144LL);
  if ( (char *)v10 == (char *)sub_3020010 )
    v11 = v9 + 960;
  else
    v11 = v10();
  switch ( a5 )
  {
    case 8173:
    case 8174:
    case 8175:
    case 8176:
    case 8177:
    case 8178:
    case 8179:
    case 8180:
    case 8193:
    case 8194:
    case 8195:
    case 8196:
    case 8197:
    case 8198:
    case 8199:
    case 8200:
    case 8201:
    case 8202:
    case 8203:
    case 8204:
    case 8205:
    case 8206:
    case 8225:
    case 8226:
      v24 = sub_B43CC0(a3);
      *(_DWORD *)a2 = 47;
      *(_DWORD *)(a2 + 8) = sub_2D5BAE0((__int64)a1, v24, *(__int64 **)(a3 + 8), 0);
      *(_QWORD *)(a2 + 16) = v25;
      v26 = -(__int64)(*(_DWORD *)(a3 + 4) & 0x7FFFFFF);
      goto LABEL_28;
    case 8181:
    case 8182:
    case 8183:
    case 8184:
    case 8208:
    case 8209:
    case 8210:
    case 8211:
    case 8212:
    case 8915:
    case 8916:
    case 8917:
    case 8918:
      v32 = sub_B43CA0(a3);
      *(_DWORD *)a2 = 47;
      *(_DWORD *)(a2 + 8) = sub_2D5BAE0((__int64)a1, v32 + 312, *(__int64 **)(a3 + 8), 0);
      *(_QWORD *)(a2 + 16) = v33;
      v26 = 1LL - (*(_DWORD *)(a3 + 4) & 0x7FFFFFF);
      goto LABEL_28;
    case 8215:
    case 8216:
      *(_DWORD *)a2 = 47;
      *(_QWORD *)(a2 + 16) = 0;
      *(_WORD *)(a2 + 8) = 2 * (a5 != 8215) + 147;
      v26 = 1LL - (*(_DWORD *)(a3 + 4) & 0x7FFFFFF);
LABEL_28:
      v27 = *(_QWORD *)(a3 + 32 * v26);
      goto LABEL_29;
    case 8276:
    case 8277:
    case 8278:
    case 8829:
    case 8830:
    case 8831:
    case 8832:
    case 8843:
    case 8844:
    case 8845:
    case 8846:
    case 8853:
    case 8854:
    case 8855:
    case 8856:
    case 8883:
    case 8884:
    case 8885:
    case 8886:
    case 8887:
    case 8891:
    case 8892:
    case 8893:
    case 8894:
    case 8895:
    case 8899:
    case 8900:
    case 8901:
    case 8902:
    case 8903:
    case 8907:
    case 8908:
    case 8909:
    case 8910:
    case 8911:
      *(_DWORD *)a2 = 47;
      *(_WORD *)(a2 + 8) = 7;
      goto LABEL_23;
    case 8280:
    case 8837:
    case 8838:
    case 8851:
    case 8852:
    case 8861:
    case 8862:
    case 8890:
    case 8898:
    case 8906:
    case 8914:
    case 10327:
    case 10335:
    case 10343:
      *(_DWORD *)a2 = 48;
      *(_WORD *)(a2 + 8) = 7;
      goto LABEL_37;
    case 8293:
    case 8294:
    case 8295:
    case 8296:
    case 8297:
      v36 = sub_B43CA0(a3);
      *(_DWORD *)a2 = 47;
      v15 = *(__int64 **)(a3 + 8);
      v16 = v36 + 312;
      goto LABEL_8;
    case 8298:
    case 8299:
      *(_DWORD *)a2 = 48;
      v51 = *(__int64 **)(*(_QWORD *)(a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF)) + 8LL);
      v52 = sub_B43CA0(a3);
      *(_DWORD *)(a2 + 8) = sub_2D5BAE0(v11, v52 + 312, v51, 0);
      *(_QWORD *)(a2 + 16) = v53;
      goto LABEL_38;
    case 8941:
    case 8942:
    case 8943:
    case 8949:
    case 8950:
    case 8955:
    case 8956:
    case 10785:
    case 10788:
    case 10789:
    case 10790:
    case 10793:
    case 10796:
    case 10797:
    case 10798:
    case 10801:
    case 10804:
    case 10805:
    case 10806:
    case 10809:
    case 10812:
    case 10813:
    case 10814:
    case 10927:
    case 10930:
    case 10935:
    case 10938:
    case 11019:
    case 11022:
    case 11027:
    case 11030:
    case 11131:
    case 11132:
    case 11133:
    case 11134:
    case 11149:
    case 11150:
    case 11151:
    case 11152:
      *(_DWORD *)a2 = 47;
      *(_WORD *)(a2 + 8) = 58;
      goto LABEL_18;
    case 8944:
    case 8945:
    case 8946:
    case 8951:
    case 8952:
    case 8957:
    case 8958:
    case 10783:
    case 10786:
    case 10791:
    case 10794:
    case 10799:
    case 10802:
    case 10807:
    case 10810:
    case 10891:
    case 10892:
    case 10893:
    case 10894:
    case 10895:
    case 10896:
    case 10897:
    case 10898:
    case 10913:
    case 10916:
    case 10917:
    case 10918:
    case 10921:
    case 10924:
    case 10925:
    case 10926:
    case 11037:
    case 11040:
    case 11041:
    case 11042:
    case 11045:
    case 11048:
    case 11049:
    case 11050:
      *(_DWORD *)a2 = 47;
      *(_WORD *)(a2 + 8) = 60;
      goto LABEL_20;
    case 8947:
    case 8948:
    case 8953:
    case 8954:
    case 10929:
    case 10932:
    case 10933:
    case 10934:
    case 10937:
    case 10940:
    case 10941:
    case 10942:
    case 11021:
    case 11024:
    case 11025:
    case 11026:
    case 11029:
    case 11032:
    case 11033:
    case 11034:
    case 11127:
    case 11128:
    case 11129:
    case 11130:
    case 11141:
    case 11142:
    case 11143:
    case 11144:
    case 11145:
    case 11146:
    case 11147:
    case 11148:
      *(_DWORD *)a2 = 47;
      *(_WORD *)(a2 + 8) = 7;
      *(_QWORD *)(a2 + 16) = 0;
      v23 = *(_QWORD *)(a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF));
      *(_DWORD *)(a2 + 40) = 0;
      *(_DWORD *)(a2 + 56) = 65794;
      *(_QWORD *)(a2 + 24) = v23 & 0xFFFFFFFFFFFFFFFBLL;
      return 1;
    case 8959:
    case 8960:
    case 8961:
      v37 = sub_B43CC0(a3);
      *(_DWORD *)a2 = 47;
      if ( a5 == 8961 )
      {
        v38 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*a1 + 32LL);
        if ( v38 == sub_2D42F30 )
        {
          v39 = sub_AE2980(v37, 0)[1];
          v40 = 2;
          if ( v39 != 1 )
          {
            v40 = 3;
            if ( v39 != 2 )
            {
              v40 = 4;
              if ( v39 != 4 )
              {
                v40 = 5;
                if ( v39 != 8 )
                {
                  switch ( v39 )
                  {
                    case 16:
                      v40 = 6;
                      break;
                    case 32:
                      v40 = 7;
                      break;
                    case 64:
                      v40 = 8;
                      break;
                    default:
                      v40 = 9 * (v39 == 128);
                      break;
                  }
                }
              }
            }
          }
        }
        else
        {
          v40 = v38((__int64)a1, v37, 0);
        }
        *(_WORD *)(a2 + 8) = v40;
        *(_QWORD *)(a2 + 16) = 0;
      }
      else
      {
        *(_DWORD *)(a2 + 8) = sub_2D5BAE0((__int64)a1, v37, *(__int64 **)(a3 + 8), 0);
        *(_QWORD *)(a2 + 16) = v66;
      }
      v41 = *(_QWORD *)(a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF));
      *(_DWORD *)(a2 + 40) = 0;
      *(_WORD *)(a2 + 58) = 1;
      *(_QWORD *)(a2 + 24) = v41 & 0xFFFFFFFFFFFFFFFBLL;
      v42 = *(_QWORD *)(a3 + 32 * (1LL - (*(_DWORD *)(a3 + 4) & 0x7FFFFFF)));
      v43 = *(_DWORD *)(v42 + 32) <= 0x40u;
      v44 = *(_QWORD *)(v42 + 24);
      if ( !v43 )
        v44 = *(_QWORD *)v44;
      v45 = 0;
      if ( v44 )
      {
        _BitScanReverse64(&v44, v44);
        v45 = 1;
        v70 = 63 - (v44 ^ 0x3F);
      }
      *(_BYTE *)(a2 + 57) = v45;
      *(_BYTE *)(a2 + 56) = v70;
      return 1;
    case 8985:
    case 8986:
    case 8987:
      *(_DWORD *)a2 = 47;
      *(_WORD *)(a2 + 8) = 7;
      goto LABEL_48;
    case 9008:
      *(_DWORD *)a2 = 47;
      result = 1;
      *(_WORD *)(a2 + 8) = 2;
      *(_QWORD *)(a2 + 16) = 0;
      *(_WORD *)(a2 + 58) = 3;
      return result;
    case 9056:
    case 9059:
      *(_DWORD *)a2 = 47;
      v63 = *(__int64 **)(*(_QWORD *)(a3 + 32 * (2LL - (*(_DWORD *)(a3 + 4) & 0x7FFFFFF))) + 8LL);
      v64 = sub_B43CA0(a3);
      *(_DWORD *)(a2 + 8) = sub_2D5BAE0(v11, v64 + 312, v63, 0);
      *(_QWORD *)(a2 + 16) = v65;
      v27 = *(_QWORD *)(a3 + 32 * (2LL - (*(_DWORD *)(a3 + 4) & 0x7FFFFFF)));
LABEL_29:
      v18 = *(_BYTE *)(a2 + 57) == 0;
      *(_DWORD *)(a2 + 40) = 0;
      *(_QWORD *)(a2 + 24) = v27 & 0xFFFFFFFFFFFFFFFBLL;
      *(_WORD *)(a2 + 58) = 3;
      if ( v18 )
        goto LABEL_25;
      goto LABEL_10;
    case 9067:
    case 9145:
      *(_DWORD *)a2 = 47;
      *(_DWORD *)(a2 + 40) = 0;
      if ( a5 == 9067 )
      {
        *(_QWORD *)(a2 + 24) = *(_QWORD *)(a3 + 32 * (1LL - (*(_DWORD *)(a3 + 4) & 0x7FFFFFF))) & 0xFFFFFFFFFFFFFFFBLL;
        v67 = *(__int64 **)(a3 + 8);
        v68 = sub_B43CA0(a3);
        v60 = 1;
        *(_DWORD *)(a2 + 8) = sub_2D5BAE0(v11, v68 + 312, v67, 0);
        v62 = 1;
        *(_QWORD *)(a2 + 16) = v69;
      }
      else
      {
        *(_QWORD *)(a2 + 24) = *(_QWORD *)(a3 + 32 * (2LL - (*(_DWORD *)(a3 + 4) & 0x7FFFFFF))) & 0xFFFFFFFFFFFFFFFBLL;
        v57 = *(__int64 **)(*(_QWORD *)(a3 + 32 * (1LL - (*(_DWORD *)(a3 + 4) & 0x7FFFFFF))) + 8LL);
        v58 = sub_B43CA0(a3);
        v59 = sub_2D5BAE0(v11, v58 + 312, v57, 0);
        v60 = 2;
        *(_QWORD *)(a2 + 16) = v61;
        *(_DWORD *)(a2 + 8) = v59;
        v62 = 2;
      }
      *(_WORD *)(a2 + 58) = v62;
      *(_WORD *)(a2 + 56) = sub_A74840((_QWORD *)(a3 + 72), v60);
      return 1;
    case 9381:
    case 9382:
      *(_DWORD *)a2 = 47;
      *(_DWORD *)(a2 + 40) = 0;
      *(_QWORD *)(a2 + 24) = *(_QWORD *)(a3 + 32 * (1LL - (*(_DWORD *)(a3 + 4) & 0x7FFFFFF))) & 0xFFFFFFFFFFFFFFFBLL;
      v54 = *(__int64 **)(*(_QWORD *)(a3 + 32 * (2LL - (*(_DWORD *)(a3 + 4) & 0x7FFFFFF))) + 8LL);
      v55 = sub_B43CA0(a3);
      *(_DWORD *)(a2 + 8) = sub_2D5BAE0(v11, v55 + 312, v54, 0);
      *(_QWORD *)(a2 + 16) = v56;
      *(_WORD *)(a2 + 58) = 3;
      *(_WORD *)(a2 + 56) = sub_A74840((_QWORD *)(a3 + 72), 1);
      return 1;
    case 9480:
    case 9490:
    case 9512:
      *(_DWORD *)a2 = 47;
      v47 = **(__int64 ***)(*(_QWORD *)(a3 + 8) + 16LL);
      v48 = sub_B43CA0(a3);
      v49 = sub_2D5BAE0(v11, v48 + 312, v47, 0);
      *(_QWORD *)(a2 + 24) = 0;
      *(_DWORD *)(a2 + 8) = v49;
      *(_QWORD *)(a2 + 16) = v50;
      *(_DWORD *)(a2 + 40) = 0;
      *(_DWORD *)(a2 + 56) = 65796;
      return 1;
    case 9556:
    case 9557:
      *(_DWORD *)a2 = 47;
      *(_WORD *)(a2 + 8) = 7;
      *(_QWORD *)(a2 + 16) = 0;
      goto LABEL_9;
    case 9558:
    case 9561:
    case 9562:
      *(_DWORD *)a2 = 48;
      *(_WORD *)(a2 + 8) = 7;
      *(_QWORD *)(a2 + 16) = 0;
      v46 = *(_QWORD *)(a3
                      + 32
                      * ((*(_DWORD *)(a3 + 4) & 0x7FFFFFFu) - 1 - (unsigned __int64)(*(_DWORD *)(a3 + 4) & 0x7FFFFFF)));
      *(_DWORD *)(a2 + 40) = 0;
      *(_DWORD *)(a2 + 56) = 131330;
      *(_QWORD *)(a2 + 24) = v46 & 0xFFFFFFFFFFFFFFFBLL;
      return 1;
    case 9559:
    case 9563:
    case 9564:
      *(_DWORD *)a2 = 48;
      *(_WORD *)(a2 + 8) = 58;
      *(_QWORD *)(a2 + 16) = 0;
      v34 = (*(_DWORD *)(a3 + 4) & 0x7FFFFFFu) - 1 - (unsigned __int64)(*(_DWORD *)(a3 + 4) & 0x7FFFFFF);
      goto LABEL_56;
    case 9560:
    case 9565:
    case 9566:
      *(_DWORD *)a2 = 48;
      *(_WORD *)(a2 + 8) = 60;
      *(_QWORD *)(a2 + 16) = 0;
      v28 = (*(_DWORD *)(a3 + 4) & 0x7FFFFFFu) - 1 - (unsigned __int64)(*(_DWORD *)(a3 + 4) & 0x7FFFFFF);
      goto LABEL_34;
    case 9598:
    case 9599:
    case 9600:
    case 9610:
    case 9611:
    case 9612:
    case 9622:
    case 9623:
    case 9624:
    case 9631:
    case 9632:
    case 9633:
    case 9643:
    case 9644:
    case 9645:
    case 9655:
    case 9656:
    case 9657:
    case 9664:
    case 9665:
    case 9666:
    case 9676:
    case 9677:
    case 9678:
    case 9688:
    case 9689:
    case 9690:
    case 9697:
    case 9698:
    case 9699:
    case 9709:
    case 9710:
    case 9711:
    case 9721:
    case 9722:
    case 9723:
    case 9730:
    case 9731:
    case 9732:
    case 9742:
    case 9743:
    case 9744:
    case 9754:
    case 9755:
    case 9756:
    case 9763:
    case 9764:
    case 9765:
    case 9775:
    case 9776:
    case 9777:
    case 9787:
    case 9788:
    case 9789:
      *(_DWORD *)a2 = 47;
      *(_WORD *)(a2 + 8) = 6;
      goto LABEL_12;
    case 9601:
    case 9602:
    case 9603:
    case 9613:
    case 9614:
    case 9615:
    case 9625:
    case 9626:
    case 9627:
    case 9634:
    case 9635:
    case 9636:
    case 9646:
    case 9647:
    case 9648:
    case 9658:
    case 9659:
    case 9660:
    case 9667:
    case 9668:
    case 9669:
    case 9679:
    case 9680:
    case 9681:
    case 9691:
    case 9692:
    case 9693:
    case 9700:
    case 9701:
    case 9702:
    case 9712:
    case 9713:
    case 9714:
    case 9724:
    case 9725:
    case 9726:
    case 9733:
    case 9734:
    case 9735:
    case 9745:
    case 9746:
    case 9747:
    case 9757:
    case 9758:
    case 9759:
    case 9766:
    case 9767:
    case 9768:
    case 9778:
    case 9779:
    case 9780:
    case 9790:
    case 9791:
    case 9792:
      *(_DWORD *)a2 = 47;
      *(_WORD *)(a2 + 8) = 7;
      goto LABEL_12;
    case 9604:
    case 9605:
    case 9606:
    case 9616:
    case 9617:
    case 9618:
    case 9637:
    case 9638:
    case 9639:
    case 9649:
    case 9650:
    case 9651:
    case 9670:
    case 9671:
    case 9672:
    case 9682:
    case 9683:
    case 9684:
    case 9703:
    case 9704:
    case 9705:
    case 9715:
    case 9716:
    case 9717:
    case 9736:
    case 9737:
    case 9738:
    case 9748:
    case 9749:
    case 9750:
    case 9769:
    case 9770:
    case 9771:
    case 9781:
    case 9782:
    case 9783:
      *(_DWORD *)a2 = 47;
      *(_WORD *)(a2 + 8) = 8;
      goto LABEL_12;
    case 9607:
    case 9608:
    case 9609:
    case 9619:
    case 9620:
    case 9621:
    case 9628:
    case 9629:
    case 9630:
    case 9640:
    case 9641:
    case 9642:
    case 9652:
    case 9653:
    case 9654:
    case 9661:
    case 9662:
    case 9663:
    case 9673:
    case 9674:
    case 9675:
    case 9685:
    case 9686:
    case 9687:
    case 9694:
    case 9695:
    case 9696:
    case 9706:
    case 9707:
    case 9708:
    case 9718:
    case 9719:
    case 9720:
    case 9727:
    case 9728:
    case 9729:
    case 9739:
    case 9740:
    case 9741:
    case 9751:
    case 9752:
    case 9753:
    case 9760:
    case 9761:
    case 9762:
    case 9772:
    case 9773:
    case 9774:
    case 9784:
    case 9785:
    case 9786:
    case 9793:
    case 9794:
    case 9795:
      *(_DWORD *)a2 = 47;
      *(_WORD *)(a2 + 8) = 5;
      goto LABEL_12;
    case 10080:
    case 10083:
    case 10089:
    case 10090:
    case 10091:
    case 10092:
    case 10101:
    case 10140:
    case 10143:
    case 10311:
    case 10351:
      *(_DWORD *)a2 = 48;
      result = 1;
      *(_WORD *)(a2 + 8) = 2;
      *(_QWORD *)(a2 + 16) = 0;
      *(_WORD *)(a2 + 58) = 3;
      return result;
    case 10146:
    case 10162:
    case 10170:
    case 10178:
    case 10264:
    case 10265:
    case 10285:
    case 10286:
      *(_DWORD *)a2 = 47;
      *(_WORD *)(a2 + 8) = 58;
      goto LABEL_48;
    case 10147:
    case 10158:
    case 10163:
    case 10171:
    case 10179:
    case 10267:
    case 10268:
    case 10288:
    case 10289:
      *(_DWORD *)a2 = 47;
      *(_WORD *)(a2 + 8) = 78;
      goto LABEL_48;
    case 10148:
    case 10153:
    case 10164:
    case 10172:
    case 10180:
    case 10270:
    case 10271:
    case 10291:
    case 10292:
      *(_DWORD *)a2 = 47;
      *(_WORD *)(a2 + 8) = 60;
      goto LABEL_48;
    case 10149:
    case 10154:
    case 10165:
    case 10173:
    case 10181:
    case 10273:
    case 10274:
    case 10294:
    case 10295:
      *(_DWORD *)a2 = 47;
      *(_WORD *)(a2 + 8) = 94;
      goto LABEL_48;
    case 10150:
    case 10155:
    case 10166:
    case 10174:
    case 10182:
    case 10276:
    case 10277:
    case 10297:
    case 10298:
      *(_DWORD *)a2 = 47;
      *(_WORD *)(a2 + 8) = 64;
      goto LABEL_48;
    case 10151:
    case 10156:
    case 10160:
    case 10168:
    case 10176:
    case 10258:
    case 10259:
    case 10279:
    case 10280:
      *(_DWORD *)a2 = 47;
      *(_WORD *)(a2 + 8) = 110;
      goto LABEL_48;
    case 10152:
    case 10157:
    case 10161:
    case 10169:
    case 10177:
    case 10261:
    case 10262:
    case 10282:
    case 10283:
      *(_DWORD *)a2 = 47;
      *(_WORD *)(a2 + 8) = 70;
      goto LABEL_48;
    case 10183:
    case 10199:
    case 10207:
    case 10215:
      *(_DWORD *)a2 = 48;
      *(_WORD *)(a2 + 8) = 58;
      goto LABEL_59;
    case 10184:
    case 10195:
    case 10200:
    case 10208:
    case 10216:
      *(_DWORD *)a2 = 48;
      *(_WORD *)(a2 + 8) = 78;
      goto LABEL_59;
    case 10185:
    case 10190:
    case 10201:
    case 10209:
    case 10217:
      *(_DWORD *)a2 = 48;
      *(_WORD *)(a2 + 8) = 60;
      goto LABEL_59;
    case 10186:
    case 10191:
    case 10202:
    case 10210:
    case 10218:
      *(_DWORD *)a2 = 48;
      *(_WORD *)(a2 + 8) = 94;
      goto LABEL_59;
    case 10187:
    case 10192:
    case 10203:
    case 10211:
    case 10219:
      *(_DWORD *)a2 = 48;
      *(_WORD *)(a2 + 8) = 64;
      goto LABEL_59;
    case 10188:
    case 10193:
    case 10197:
    case 10205:
    case 10213:
      *(_DWORD *)a2 = 48;
      *(_WORD *)(a2 + 8) = 110;
      goto LABEL_59;
    case 10189:
    case 10194:
    case 10198:
    case 10206:
    case 10214:
      *(_DWORD *)a2 = 48;
      *(_WORD *)(a2 + 8) = 70;
      goto LABEL_59;
    case 10196:
    case 10204:
    case 10212:
      *(_DWORD *)a2 = 48;
      *(_WORD *)(a2 + 8) = 7;
LABEL_59:
      *(_QWORD *)(a2 + 16) = 0;
      v30 = 1LL - (*(_DWORD *)(a3 + 4) & 0x7FFFFFF);
      goto LABEL_39;
    case 10220:
    case 10236:
    case 10244:
    case 10252:
      *(_DWORD *)a2 = 47;
      *(_WORD *)(a2 + 8) = 58;
      goto LABEL_23;
    case 10221:
    case 10232:
    case 10237:
    case 10245:
    case 10253:
      *(_DWORD *)a2 = 47;
      *(_WORD *)(a2 + 8) = 78;
      goto LABEL_23;
    case 10222:
    case 10227:
    case 10238:
    case 10246:
    case 10254:
      *(_DWORD *)a2 = 47;
      *(_WORD *)(a2 + 8) = 60;
      goto LABEL_23;
    case 10223:
    case 10228:
    case 10239:
    case 10247:
    case 10255:
      *(_DWORD *)a2 = 47;
      *(_WORD *)(a2 + 8) = 94;
      goto LABEL_23;
    case 10224:
    case 10229:
    case 10240:
    case 10248:
    case 10256:
      *(_DWORD *)a2 = 47;
      *(_WORD *)(a2 + 8) = 64;
      goto LABEL_23;
    case 10225:
    case 10230:
    case 10234:
    case 10242:
    case 10250:
      *(_DWORD *)a2 = 47;
      *(_WORD *)(a2 + 8) = 110;
      goto LABEL_23;
    case 10226:
    case 10231:
    case 10235:
    case 10243:
    case 10251:
      *(_DWORD *)a2 = 47;
      *(_WORD *)(a2 + 8) = 70;
      goto LABEL_23;
    case 10233:
    case 10241:
    case 10249:
      *(_DWORD *)a2 = 47;
      *(_WORD *)(a2 + 8) = 57;
LABEL_23:
      *(_QWORD *)(a2 + 16) = 0;
      v21 = -(__int64)(*(_DWORD *)(a3 + 4) & 0x7FFFFFF);
      goto LABEL_24;
    case 10257:
    case 10278:
      *(_DWORD *)a2 = 47;
      *(_WORD *)(a2 + 8) = 161;
      goto LABEL_48;
    case 10260:
    case 10281:
      *(_DWORD *)a2 = 47;
      *(_WORD *)(a2 + 8) = 158;
      goto LABEL_48;
    case 10263:
    case 10284:
      *(_DWORD *)a2 = 47;
      *(_WORD *)(a2 + 8) = 147;
      goto LABEL_48;
    case 10266:
    case 10287:
      *(_DWORD *)a2 = 47;
      *(_WORD *)(a2 + 8) = 159;
      goto LABEL_48;
    case 10269:
    case 10290:
      *(_DWORD *)a2 = 47;
      *(_WORD *)(a2 + 8) = 149;
      goto LABEL_48;
    case 10272:
    case 10293:
      *(_DWORD *)a2 = 47;
      *(_WORD *)(a2 + 8) = 160;
      goto LABEL_48;
    case 10275:
    case 10296:
      *(_DWORD *)a2 = 47;
      *(_WORD *)(a2 + 8) = 153;
LABEL_48:
      *(_QWORD *)(a2 + 16) = 0;
      v21 = 1LL - (*(_DWORD *)(a3 + 4) & 0x7FFFFFF);
LABEL_24:
      v22 = *(_QWORD *)(a3 + 32 * v21);
      *(_DWORD *)(a2 + 40) = 0;
      *(_WORD *)(a2 + 58) = 1;
      v18 = *(_BYTE *)(a2 + 57) == 0;
      *(_QWORD *)(a2 + 24) = v22 & 0xFFFFFFFFFFFFFFFBLL;
      if ( v18 )
        goto LABEL_25;
      goto LABEL_10;
    case 10299:
    case 10300:
    case 10301:
    case 10302:
    case 10303:
    case 10304:
    case 10305:
    case 10306:
    case 10307:
    case 10308:
      *(_DWORD *)a2 = 47;
      result = 1;
      *(_WORD *)(a2 + 8) = 7;
      *(_QWORD *)(a2 + 16) = 0;
      *(_QWORD *)(a2 + 24) = 0;
      *(_DWORD *)(a2 + 40) = 0;
      *(_DWORD *)(a2 + 56) = 196868;
      return result;
    case 10314:
    case 10330:
    case 10338:
    case 10346:
      *(_DWORD *)a2 = 48;
      *(_WORD *)(a2 + 8) = 58;
      goto LABEL_37;
    case 10315:
    case 10326:
    case 10331:
    case 10339:
    case 10347:
      *(_DWORD *)a2 = 48;
      *(_WORD *)(a2 + 8) = 78;
      goto LABEL_37;
    case 10316:
    case 10321:
    case 10332:
    case 10340:
    case 10348:
      *(_DWORD *)a2 = 48;
      *(_WORD *)(a2 + 8) = 60;
      goto LABEL_37;
    case 10317:
    case 10322:
    case 10333:
    case 10341:
    case 10349:
      *(_DWORD *)a2 = 48;
      *(_WORD *)(a2 + 8) = 94;
      goto LABEL_37;
    case 10318:
    case 10323:
    case 10334:
    case 10342:
    case 10350:
      *(_DWORD *)a2 = 48;
      *(_WORD *)(a2 + 8) = 64;
      goto LABEL_37;
    case 10319:
    case 10324:
    case 10328:
    case 10336:
    case 10344:
      *(_DWORD *)a2 = 48;
      *(_WORD *)(a2 + 8) = 110;
      goto LABEL_37;
    case 10320:
    case 10325:
    case 10329:
    case 10337:
    case 10345:
      *(_DWORD *)a2 = 48;
      *(_WORD *)(a2 + 8) = 70;
LABEL_37:
      *(_QWORD *)(a2 + 16) = 0;
LABEL_38:
      v30 = -(__int64)(*(_DWORD *)(a3 + 4) & 0x7FFFFFF);
LABEL_39:
      v31 = *(_QWORD *)(a3 + 32 * v30);
      *(_DWORD *)(a2 + 40) = 0;
      *(_WORD *)(a2 + 58) = 2;
      v18 = *(_BYTE *)(a2 + 57) == 0;
      *(_QWORD *)(a2 + 24) = v31 & 0xFFFFFFFFFFFFFFFBLL;
      if ( v18 )
        goto LABEL_25;
      goto LABEL_10;
    case 10358:
    case 10361:
    case 10364:
    case 10365:
    case 10370:
    case 10373:
    case 10376:
    case 10377:
    case 10382:
    case 10385:
    case 10388:
    case 10389:
    case 10394:
    case 10397:
    case 10400:
    case 10401:
    case 10406:
    case 10409:
    case 10412:
    case 10413:
    case 10418:
    case 10421:
    case 10424:
    case 10427:
    case 10500:
    case 10503:
    case 10506:
    case 10507:
    case 10512:
    case 10515:
    case 10518:
    case 10519:
    case 10524:
    case 10527:
    case 10530:
    case 10531:
    case 10536:
    case 10539:
    case 10542:
    case 10543:
    case 10548:
    case 10551:
    case 10554:
    case 10555:
    case 10560:
    case 10563:
    case 10566:
    case 10569:
    case 10572:
    case 10575:
    case 10580:
    case 10583:
    case 10586:
    case 10589:
    case 10592:
    case 10595:
    case 10598:
    case 10601:
      *(_DWORD *)a2 = 47;
      *(_WORD *)(a2 + 8) = 149;
      goto LABEL_12;
    case 10359:
    case 10360:
    case 10362:
    case 10363:
    case 10366:
    case 10367:
    case 10368:
    case 10369:
    case 10371:
    case 10372:
    case 10374:
    case 10375:
    case 10378:
    case 10379:
    case 10380:
    case 10381:
    case 10383:
    case 10384:
    case 10386:
    case 10387:
    case 10390:
    case 10391:
    case 10392:
    case 10393:
    case 10395:
    case 10396:
    case 10398:
    case 10399:
    case 10402:
    case 10403:
    case 10404:
    case 10405:
    case 10407:
    case 10408:
    case 10410:
    case 10411:
    case 10414:
    case 10415:
    case 10416:
    case 10417:
    case 10419:
    case 10420:
    case 10422:
    case 10423:
    case 10425:
    case 10426:
    case 10428:
    case 10429:
    case 10501:
    case 10502:
    case 10504:
    case 10505:
    case 10508:
    case 10509:
    case 10510:
    case 10511:
    case 10513:
    case 10514:
    case 10516:
    case 10517:
    case 10520:
    case 10521:
    case 10522:
    case 10523:
    case 10525:
    case 10526:
    case 10528:
    case 10529:
    case 10532:
    case 10533:
    case 10534:
    case 10535:
    case 10537:
    case 10538:
    case 10540:
    case 10541:
    case 10544:
    case 10545:
    case 10546:
    case 10547:
    case 10549:
    case 10550:
    case 10552:
    case 10553:
    case 10556:
    case 10557:
    case 10558:
    case 10559:
    case 10561:
    case 10562:
    case 10564:
    case 10565:
    case 10567:
    case 10568:
    case 10570:
    case 10571:
    case 10573:
    case 10574:
    case 10576:
    case 10577:
    case 10581:
    case 10582:
    case 10584:
    case 10585:
    case 10587:
    case 10588:
    case 10590:
    case 10591:
    case 10593:
    case 10594:
    case 10596:
    case 10597:
    case 10599:
    case 10600:
    case 10602:
    case 10603:
      *(_DWORD *)a2 = 47;
      *(_WORD *)(a2 + 8) = 60;
LABEL_12:
      *(_QWORD *)(a2 + 16) = 0;
      *(_QWORD *)(a2 + 24) = 0;
      *(_DWORD *)(a2 + 40) = 0;
      *(_DWORD *)(a2 + 56) = 65796;
      return 1;
    case 10654:
    case 10655:
    case 10656:
    case 10657:
    case 10658:
    case 10659:
    case 10660:
    case 10661:
    case 10662:
    case 10663:
    case 10664:
    case 10665:
    case 10666:
    case 10667:
    case 10668:
    case 10669:
    case 10670:
    case 10671:
    case 10672:
    case 10673:
    case 10674:
    case 10675:
    case 10676:
    case 10677:
    case 10678:
    case 10679:
    case 10680:
    case 10681:
    case 10682:
    case 10683:
    case 10684:
    case 10685:
    case 10686:
    case 10687:
    case 10688:
    case 10689:
    case 10690:
    case 10691:
    case 10692:
    case 10693:
    case 10694:
    case 10695:
    case 10696:
    case 10697:
    case 10698:
    case 10699:
    case 10700:
    case 10701:
    case 10702:
    case 10703:
    case 10704:
    case 10705:
    case 10706:
    case 10707:
    case 10708:
    case 10709:
    case 10710:
    case 10711:
    case 10712:
    case 10713:
    case 10714:
    case 10715:
    case 10716:
    case 10717:
    case 10718:
    case 10719:
    case 10720:
    case 10721:
    case 10722:
    case 10723:
    case 10724:
    case 10725:
    case 10726:
    case 10727:
    case 10728:
    case 10729:
    case 10730:
    case 10731:
    case 10732:
    case 10733:
    case 10734:
    case 10735:
    case 10736:
    case 10737:
    case 10738:
    case 10739:
    case 10740:
    case 10741:
    case 10742:
    case 10743:
    case 10744:
    case 10745:
    case 10746:
    case 10747:
    case 10748:
    case 10749:
    case 10750:
    case 10751:
    case 10752:
    case 10753:
    case 10754:
    case 10755:
    case 10756:
    case 10757:
    case 10758:
    case 10759:
    case 10760:
    case 10761:
    case 10762:
    case 10763:
    case 10764:
    case 10765:
    case 10766:
    case 10767:
    case 10768:
    case 10769:
    case 10770:
    case 10771:
    case 10772:
    case 10773:
    case 10774:
    case 10775:
    case 10776:
    case 10777:
    case 10778:
    case 10779:
    case 10780:
    case 10781:
      *(_DWORD *)a2 = 47;
      v13 = *(__int64 **)(a3 + 8);
      v14 = sub_B43CA0(a3);
      v15 = v13;
      v16 = v14 + 312;
LABEL_8:
      *(_DWORD *)(a2 + 8) = sub_2D5BAE0((__int64)a1, v16, v15, 0);
      *(_QWORD *)(a2 + 16) = v17;
LABEL_9:
      v18 = *(_BYTE *)(a2 + 57) == 0;
      *(_QWORD *)(a2 + 24) = 0;
      *(_DWORD *)(a2 + 40) = 0;
      *(_WORD *)(a2 + 58) = 1;
      if ( v18 )
        goto LABEL_25;
LABEL_10:
      *(_BYTE *)(a2 + 57) = 0;
      result = 1;
      break;
    case 10784:
    case 10787:
    case 10792:
    case 10795:
    case 10800:
    case 10803:
    case 10808:
    case 10811:
    case 10912:
    case 10915:
    case 10920:
    case 10923:
    case 10928:
    case 10931:
    case 10936:
    case 10939:
    case 11020:
    case 11023:
    case 11028:
    case 11031:
    case 11036:
    case 11039:
    case 11044:
    case 11047:
      *(_DWORD *)a2 = 47;
      *(_WORD *)(a2 + 8) = 130;
      goto LABEL_20;
    case 10815:
    case 10818:
    case 10821:
    case 10824:
    case 10943:
    case 10946:
    case 10949:
    case 10952:
    case 11051:
    case 11054:
    case 11057:
    case 11060:
      *(_DWORD *)a2 = 47;
      *(_WORD *)(a2 + 8) = 129;
      goto LABEL_20;
    case 10816:
    case 10819:
    case 10822:
    case 10825:
    case 10899:
    case 10900:
    case 10901:
    case 10902:
    case 10944:
    case 10947:
    case 10950:
    case 10953:
    case 11052:
    case 11055:
    case 11058:
    case 11061:
      *(_DWORD *)a2 = 47;
      *(_WORD *)(a2 + 8) = 153;
      goto LABEL_20;
    case 10817:
    case 10820:
    case 10823:
    case 10826:
    case 10911:
    case 10914:
    case 10919:
    case 10922:
    case 10945:
    case 10948:
    case 10951:
    case 10954:
    case 11035:
    case 11038:
    case 11043:
    case 11046:
    case 11053:
    case 11056:
    case 11059:
    case 11062:
      *(_DWORD *)a2 = 47;
      *(_WORD *)(a2 + 8) = 64;
      goto LABEL_20;
    case 10879:
    case 10882:
    case 10885:
    case 10888:
    case 11007:
    case 11010:
    case 11013:
    case 11016:
    case 11115:
    case 11118:
    case 11121:
    case 11124:
      *(_DWORD *)a2 = 48;
      *(_WORD *)(a2 + 8) = 129;
      *(_QWORD *)(a2 + 16) = 0;
      v28 = -(__int64)(*(_DWORD *)(a3 + 4) & 0x7FFFFFF);
      goto LABEL_34;
    case 10880:
    case 10883:
    case 10886:
    case 10889:
    case 10907:
    case 10908:
    case 10909:
    case 10910:
    case 11008:
    case 11011:
    case 11014:
    case 11017:
    case 11116:
    case 11119:
    case 11122:
    case 11125:
      *(_DWORD *)a2 = 48;
      *(_WORD *)(a2 + 8) = 153;
      *(_QWORD *)(a2 + 16) = 0;
      v28 = -(__int64)(*(_DWORD *)(a3 + 4) & 0x7FFFFFF);
      goto LABEL_34;
    case 10881:
    case 10884:
    case 10887:
    case 10890:
    case 11009:
    case 11012:
    case 11015:
    case 11018:
    case 11117:
    case 11120:
    case 11123:
    case 11126:
      *(_DWORD *)a2 = 48;
      *(_WORD *)(a2 + 8) = 64;
      *(_QWORD *)(a2 + 16) = 0;
      v28 = -(__int64)(*(_DWORD *)(a3 + 4) & 0x7FFFFFF);
      goto LABEL_34;
    case 11137:
    case 11138:
    case 11139:
    case 11140:
    case 11157:
    case 11158:
    case 11159:
    case 11160:
      *(_DWORD *)a2 = 48;
      *(_WORD *)(a2 + 8) = 58;
      *(_QWORD *)(a2 + 16) = 0;
      v34 = -(__int64)(*(_DWORD *)(a3 + 4) & 0x7FFFFFF);
LABEL_56:
      v35 = *(_QWORD *)(a3 + 32 * v34);
      *(_DWORD *)(a2 + 40) = 0;
      *(_DWORD *)(a2 + 56) = 131331;
      *(_QWORD *)(a2 + 24) = v35 & 0xFFFFFFFFFFFFFFFBLL;
      goto LABEL_25;
    case 11161:
    case 11162:
    case 11163:
    case 11164:
    case 11165:
    case 11166:
    case 11167:
    case 11168:
      *(_DWORD *)a2 = 47;
      *(_WORD *)(a2 + 8) = 13;
LABEL_18:
      *(_QWORD *)(a2 + 16) = 0;
      v19 = *(_QWORD *)(a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF));
      *(_DWORD *)(a2 + 40) = 0;
      *(_DWORD *)(a2 + 56) = 65795;
      *(_QWORD *)(a2 + 24) = v19 & 0xFFFFFFFFFFFFFFFBLL;
      return 1;
    case 11169:
    case 11170:
    case 11171:
    case 11172:
      *(_DWORD *)a2 = 47;
      *(_WORD *)(a2 + 8) = 167;
LABEL_20:
      *(_QWORD *)(a2 + 16) = 0;
      v20 = *(_QWORD *)(a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF));
      *(_DWORD *)(a2 + 40) = 0;
      *(_DWORD *)(a2 + 56) = 65796;
      *(_QWORD *)(a2 + 24) = v20 & 0xFFFFFFFFFFFFFFFBLL;
      result = 1;
      break;
    case 11193:
    case 11194:
    case 11195:
    case 11196:
      *(_DWORD *)a2 = 48;
      *(_WORD *)(a2 + 8) = 167;
      *(_QWORD *)(a2 + 16) = 0;
      v28 = -(__int64)(*(_DWORD *)(a3 + 4) & 0x7FFFFFF);
LABEL_34:
      v29 = *(_QWORD *)(a3 + 32 * v28);
      *(_DWORD *)(a2 + 40) = 0;
      *(_DWORD *)(a2 + 56) = 131332;
      *(_QWORD *)(a2 + 24) = v29 & 0xFFFFFFFFFFFFFFFBLL;
LABEL_25:
      result = 1;
      break;
    default:
      result = 0;
      break;
  }
  return result;
}
